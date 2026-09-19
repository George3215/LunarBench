#include "MoonViewportLibrary.h"
#include "MoonPaths.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "EngineUtils.h"
#include "HAL/FileManager.h"
#include "UObject/StrongObjectPtr.h"
#include "Editor.h"
#include "EditorViewportClient.h"
#include "LevelEditorViewport.h"
#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Engine/World.h"
#include "SceneView.h"
#include "UnrealClient.h"
#include "Engine/StaticMesh.h"
#include "StaticMeshAttributes.h"
#include "MeshDescription.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
TWeakObjectPtr<ACameraActor> Camera;
TWeakObjectPtr<ASceneCapture2D> RgbSensor, DepthSensor;
TStrongObjectPtr<UTextureRenderTarget2D> RgbTarget, DepthTarget;
FLevelEditorViewportClient* Client()
{
    if(!GEditor)return nullptr;
    for(FLevelEditorViewportClient* C:GEditor->GetLevelViewportClients())
        if(C && C->IsPerspective() && C->GetWorld() && MoonPaths::IsGo2Map(C->GetWorld()->GetName()))return C;
    return nullptr;
}
}

void UMoonViewportLibrary::ApplyCamera(const TArray<double>& F)
{
    auto* C=Client();if(!C || F.Num()!=11)return;
    for(double V:F)if(!FMath::IsFinite(V))return;
    if(F[9]<=1 || F[9]>=170 || F[10]<=0)return;
    if(!Camera.IsValid() || Camera->GetWorld()!=C->GetWorld())
    {
        FActorSpawnParameters P;P.ObjectFlags=RF_Transient;P.Name=TEXT("MoonSimView");
        Camera=C->GetWorld()->SpawnActor<ACameraActor>(P);
        Camera->SetActorLabel(TEXT("MoonSim camera (temporary)"));
        Camera->GetCameraComponent()->SetConstraintAspectRatio(true);
        C->SetActorLock(Camera.Get());C->bLockedCameraView=true;
    }
    FVector Eye(F[0],F[1],F[2]),Forward(F[3],F[4],F[5]),Up(F[6],F[7],F[8]);
    FRotator Rotation=FRotationMatrix::MakeFromXZ(Forward.GetSafeNormal(),Up.GetSafeNormal()).Rotator();
    auto* Component=Camera->GetCameraComponent();
    Component->SetAspectRatio(F[10]);
    Component->SetFieldOfView(FMath::RadiansToDegrees(2*FMath::Atan(FMath::Tan(FMath::DegreesToRadians(F[9])*.5)*F[10])));
    Camera->SetActorLocationAndRotation(Eye,Rotation,false,nullptr,ETeleportType::TeleportPhysics);
    C->SetViewLocation(Eye);C->SetViewRotation(Rotation);C->ViewFOV=Component->FieldOfView;
    C->UpdateViewForLockedActor(0);
    C->Invalidate(false,false); // Keep hit proxies; do not invalidate every editor viewport.
}

TArray<double> UMoonViewportLibrary::ReadCamera()
{
    auto* C=Client();if(!C || !Camera.IsValid())return {};
    const FVector Eye=C->GetViewLocation();const FRotationMatrix R(C->GetViewRotation());
    const FVector F=R.GetUnitAxis(EAxis::X),U=R.GetUnitAxis(EAxis::Z);
    const auto* Comp=Camera->GetCameraComponent();const double A=Comp->AspectRatio;
    const double VFov=FMath::RadiansToDegrees(2*FMath::Atan(FMath::Tan(FMath::DegreesToRadians(Comp->FieldOfView)*.5)/A));
    return {Eye.X,Eye.Y,Eye.Z,F.X,F.Y,F.Z,U.X,U.Y,U.Z,VFov,A};
}

TArray<FVector> UMoonViewportLibrary::ProjectPoints(const TArray<FVector>& Points)
{
    TArray<FVector> Result;auto* C=Client();if(!C || !C->Viewport)return Result;
    FSceneViewFamilyContext Family(FSceneViewFamily::ConstructionValues(C->Viewport,C->GetScene(),C->EngineShowFlags).SetRealtimeUpdate(true));
    FSceneView* View=C->CalcSceneView(&Family);if(!View)return Result;
    for(const FVector& P:Points)
    {
        const FVector4 Clip=View->WorldToScreen(P);
        Result.Add(Clip.W>0?FVector(.5+.5*Clip.X/Clip.W,.5-.5*Clip.Y/Clip.W,Clip.W):FVector(-1,-1,Clip.W));
    }
    return Result;
}

void UMoonViewportLibrary::ReleaseCamera()
{
    if(RgbSensor.IsValid())RgbSensor->Destroy();
    if(DepthSensor.IsValid())DepthSensor->Destroy();
    RgbSensor.Reset();DepthSensor.Reset();RgbTarget.Reset();DepthTarget.Reset();
    if(auto* C=Client())if(Camera.IsValid() && C->GetActorLock().GetLockedActor()==Camera.Get())C->SetActorLock(nullptr);
    if(Camera.IsValid())Camera->Destroy();Camera.Reset();
}

// Transient mesh import: task samples share the exact MuJoCo source vertices/faces.
UStaticMesh* UMoonViewportLibrary::CreateTaskMesh(const FString& JsonFile)
{
    FString Text;
    if(!FFileHelper::LoadFileToString(Text,*JsonFile))return nullptr;
    TSharedPtr<FJsonObject> Root;
    if(!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Root))return nullptr;
    auto* Mesh=NewObject<UStaticMesh>(GetTransientPackage(),NAME_None,RF_Transient);
    Mesh->GetStaticMaterials().Add(FStaticMaterial());
    FMeshDescription D;FStaticMeshAttributes A(D);A.Register();
    auto Positions=A.GetVertexPositions();auto UV=A.GetVertexInstanceUVs();UV.SetNumChannels(1);
    for(const auto& Point:Root->GetArrayField(TEXT("points")))
    {
        const auto& P=Point->AsArray();if(P.Num()!=3)return nullptr;
        auto Id=D.CreateVertex();Positions[Id]=FVector3f(P[0]->AsNumber(),P[1]->AsNumber(),P[2]->AsNumber());
    }
    const auto& Indices=Root->GetArrayField(TEXT("indices"));auto Group=D.CreatePolygonGroup();
    if(Indices.Num()%3)return nullptr;
    // Optional per-corner UVs, parallel to "indices" (one pair per triangle corner). Rock
    // prototypes carry them so their texture lands on the right face; robot meshes omit the
    // field and keep the previous zero UVs.
    const TArray<TSharedPtr<FJsonValue>>* Uvs=nullptr;
    Root->TryGetArrayField(TEXT("uvs"),Uvs);
    if(Uvs && Uvs->Num()!=Indices.Num())return nullptr;
    for(int i=0;i<Indices.Num();i+=3)
    {
        TArray<FVertexInstanceID> Corners;
        for(int k=0;k<3;k++)
        {
            int Id=int(Indices[i+k]->AsNumber());if(Id<0 || Id>=D.Vertices().Num())return nullptr;
            auto VI=D.CreateVertexInstance(FVertexID(Id));
            FVector2f Corner(0.f,0.f);
            if(Uvs)
            {
                const auto& Pair=(*Uvs)[i+k]->AsArray();if(Pair.Num()!=2)return nullptr;
                Corner=FVector2f(Pair[0]->AsNumber(),Pair[1]->AsNumber());
            }
            UV.Set(VI,0,Corner);Corners.Add(VI);
        }
        D.CreatePolygon(Group,Corners);
    }
    auto& Source=Mesh->AddSourceModel();Source.BuildSettings.bRecomputeNormals=true;Source.BuildSettings.bRecomputeTangents=true;
    Mesh->CreateMeshDescription(0,MoveTemp(D));Mesh->CommitMeshDescription(0);Mesh->Build(false);
    return Mesh;
}

// A robot-mounted offscreen camera, independent of the editor's inspection viewport.
// RGB bytes + axial depth float32 metres; never substitutes MuJoCo depth or object poses.
bool UMoonViewportLibrary::CaptureTaskRGBD(const FString& File, FVector Position,
    FRotator Rotation, int32 Width, int32 Height, float HorizontalFov)
{
    auto* C=Client();if(!C || Width<16 || Height<16 || Width>1920 || Height>1080)return false;
    if(!RgbSensor.IsValid())
    {
        FActorSpawnParameters P;P.ObjectFlags=RF_Transient;
        RgbSensor=C->GetWorld()->SpawnActor<ASceneCapture2D>(P);
        DepthSensor=C->GetWorld()->SpawnActor<ASceneCapture2D>(P);
        RgbTarget.Reset(NewObject<UTextureRenderTarget2D>(GetTransientPackage()));
        RgbTarget->RenderTargetFormat=RTF_RGBA8;RgbTarget->InitAutoFormat(Width,Height);
        RgbTarget->UpdateResourceImmediate(true);
        DepthTarget.Reset(NewObject<UTextureRenderTarget2D>(GetTransientPackage()));
        DepthTarget->RenderTargetFormat=RTF_RGBA32f;DepthTarget->InitAutoFormat(Width,Height);
        DepthTarget->UpdateResourceImmediate(true);
        RgbSensor->GetCaptureComponent2D()->TextureTarget=RgbTarget.Get();
        DepthSensor->GetCaptureComponent2D()->TextureTarget=DepthTarget.Get();
        RgbSensor->GetCaptureComponent2D()->CaptureSource=SCS_FinalColorLDR;
        DepthSensor->GetCaptureComponent2D()->CaptureSource=SCS_SceneDepth;
        for(auto* S:{RgbSensor.Get(),DepthSensor.Get()})
        {
            auto* Capture=S->GetCaptureComponent2D();
            Capture->bCaptureEveryFrame=false;Capture->bCaptureOnMovement=false;
            Capture->bAlwaysPersistRenderingState=true;
            Capture->PostProcessSettings.bOverride_AutoExposureMethod=true;
            Capture->PostProcessSettings.AutoExposureMethod=AEM_Manual;
            Capture->PostProcessSettings.bOverride_AutoExposureApplyPhysicalCameraExposure=true;
            Capture->PostProcessSettings.AutoExposureApplyPhysicalCameraExposure=false;
            Capture->PostProcessSettings.bOverride_AutoExposureBias=true;
            Capture->PostProcessSettings.AutoExposureBias=1.f;
            for(TActorIterator<AActor> It(C->GetWorld());It;++It)
                if(It->IsTemporarilyHiddenInEditor() || It->ActorHasTag(TEXT("D435Housing")))Capture->HiddenActors.Add(*It);
        }
    }
    if(RgbTarget->SizeX!=Width || RgbTarget->SizeY!=Height)return false;
    for(auto* S:{RgbSensor.Get(),DepthSensor.Get()})
    {
        S->SetActorLocationAndRotation(Position,Rotation);
        S->GetCaptureComponent2D()->FOVAngle=HorizontalFov;
        S->GetCaptureComponent2D()->CaptureScene();
    }
    TArray<FColor> Rgb;TArray<FLinearColor> Depth;
    FReadSurfaceDataFlags Flags(RCM_MinMax);Flags.SetLinearToGamma(false);
    if(!RgbTarget->GameThread_GetRenderTargetResource()->ReadPixels(Rgb,Flags) ||
       !DepthTarget->GameThread_GetRenderTargetResource()->ReadLinearColorPixels(Depth,Flags))return false;
    const int32 N=Width*Height;if(Rgb.Num()!=N || Depth.Num()!=N)return false;
    TArray<uint8> Bytes;Bytes.SetNumUninitialized(N*7);
    for(int32 I=0;I<N;++I)
    {
        Bytes[3*I]=Rgb[I].R;Bytes[3*I+1]=Rgb[I].G;Bytes[3*I+2]=Rgb[I].B;
        float Metres=Depth[I].R*.01f;
        if(!FMath::IsFinite(Metres) || Metres<.28f || Metres>10.f)Metres=0.f;
        FMemory::Memcpy(Bytes.GetData()+N*3+I*4,&Metres,4);
    }
    FString Temporary=File+TEXT(".tmp");
    return FFileHelper::SaveArrayToFile(Bytes,*Temporary) &&
           IFileManager::Get().Move(*File,*Temporary,true,true);
}
