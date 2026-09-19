#include "Materials/MaterialExpressionCustom.h"
#include "Materials/MaterialExpressionTextureObject.h"
#include "Materials/MaterialExpressionTextureCoordinate.h"
#include "Materials/MaterialExpressionPixelDepth.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "StaticMeshAttributes.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "PhysicsEngine/BodySetup.h"

namespace MoonSceneV2
{
const FString Data = MoonPaths::File(TEXT("ue/import_data/v2/"));
const FString Dest = TEXT("/Game/MoonTerrainV2/");
TSharedPtr<FJsonObject> ReadObject(const FString& Name)
{
    FString Text; FFileHelper::LoadFileToString(Text, *(Data+Name));
    TSharedPtr<FJsonObject> Obj;
    check(FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Obj));
    return Obj;
}
FVector Vec(const TArray<TSharedPtr<FJsonValue>>& A)
{ return FVector(A[0]->AsNumber(), A[1]->AsNumber(), A[2]->AsNumber()); }
UTexture2D* Texture(const FString& Name, bool Color)
{
    FString Pkg = Dest+TEXT("Textures/")+Name;
    UTextureFactory* Factory=NewObject<UTextureFactory>();
    Factory->SuppressImportOverwriteDialog();
    Factory->ColorSpaceMode=Color ? ETextureSourceColorSpace::Auto : ETextureSourceColorSpace::Linear;
    bool Cancel=false;
    UTexture2D* T=Cast<UTexture2D>(Factory->ImportObject(UTexture2D::StaticClass(),CreatePackage(*Pkg),FName(*Name),RF_Public|RF_Standalone|RF_Transactional,
        MoonPaths::File(TEXT("assets/environments/lunar/terrain/landscape_cropped/Materials/Textures/"))+Name+TEXT(".png"),TEXT(""),Cancel));
    check(T);
    // Packed normal files also carry specular and blend height: BC5 loses B/A.
    T->CompressionSettings=TC_BC7; T->SRGB=Color;
    T->LODGroup=TEXTUREGROUP_Cinematic; T->LODBias=0;
    T->MaxTextureSize=8192; T->NeverStream=true;
    T->PostEditChange(); FAssetRegistryModule::AssetCreated(T);
    check(MoonTerrainImport::SaveAsset(T)); return T;
}
UMaterial* Material(const FString& Name, bool Rock, const TMap<FString,UTexture2D*>& Tex)
{
    UMaterialFactoryNew* Factory=NewObject<UMaterialFactoryNew>();
    UMaterial* M=CastChecked<UMaterial>(Factory->FactoryCreateNew(UMaterial::StaticClass(),CreatePackage(*(Dest+TEXT("Materials/")+Name)),FName(*Name),RF_Public|RF_Standalone,nullptr,GWarn));
    auto* C=MoonTerrainImport::CreateExpression<UMaterialExpressionCustom>(M,0,0);
    check(FFileHelper::LoadFileToString(C->Code,*(Data+(Rock ? TEXT("rock.hlsl") : TEXT("landscape.hlsl")))));
    C->OutputType=CMOT_Float3; C->Description=TEXT("USD MDL surface port: packed RG/B/A channels preserved");
    FCustomOutput N; N.OutputName=TEXT("NormalOut"); N.OutputType=CMOT_Float3; C->AdditionalOutputs.Add(N);
    FCustomOutput S; S.OutputName=TEXT("SpecOut"); S.OutputType=CMOT_Float1; C->AdditionalOutputs.Add(S);
    C->Inputs.Empty();
    auto Add=[&](FName Name,UMaterialExpression* E) { FCustomInput I; I.InputName=Name; I.Input.Connect(0,E); C->Inputs.Add(I); };
    Add(TEXT("World"),MoonTerrainImport::CreateExpression<UMaterialExpressionWorldPosition>(M,-600,0));
    Add(TEXT("Depth"),MoonTerrainImport::CreateExpression<UMaterialExpressionPixelDepth>(M,-600,120));
    Add(TEXT("UV"),MoonTerrainImport::CreateExpression<UMaterialExpressionTextureCoordinate>(M,-600,240));
    TArray<FString> Names=Rock ? TArray<FString>{TEXT("A1"),TEXT("N1"),TEXT("Mask"),TEXT("RockN")} : TArray<FString>{TEXT("A1"),TEXT("A2"),TEXT("A3"),TEXT("N1"),TEXT("N2"),TEXT("N3"),TEXT("Mask"),TEXT("Detail")};
    int Row=0;
    for(const FString& Key:Names)
    {
        auto* E=MoonTerrainImport::CreateExpression<UMaterialExpressionTextureObject>(M,-900,Row++*150);
        E->Texture=Tex[Key]; E->SamplerType=Key.StartsWith(TEXT("A")) ? SAMPLERTYPE_Color : SAMPLERTYPE_LinearColor;
        Add(FName(*Key),E);
    }
    C->RebuildOutputs();
    check(UMaterialEditingLibrary::ConnectMaterialProperty(C,TEXT(""),MP_BaseColor));
    check(UMaterialEditingLibrary::ConnectMaterialProperty(C,TEXT("NormalOut"),MP_Normal));
    check(UMaterialEditingLibrary::ConnectMaterialProperty(C,TEXT("SpecOut"),MP_Specular));
    auto* R=MoonTerrainImport::CreateExpression<UMaterialExpressionConstant>(M,0,400); R->R=1;
    UMaterialEditingLibrary::ConnectMaterialProperty(R,TEXT(""),MP_Roughness);
    UMaterialEditingLibrary::RecompileMaterial(M); M->PostEditChange();
    FAssetRegistryModule::AssetCreated(M); check(MoonTerrainImport::SaveAsset(M)); return M;
}
UStaticMesh* Mesh(int Index, UMaterial* Material)
{
    auto Obj=ReadObject(FString::Printf(TEXT("rock%d.json"),Index));
    const auto& P=Obj->GetArrayField(TEXT("points"));
    const auto& I=Obj->GetArrayField(TEXT("indices"));
    const auto& N=Obj->GetArrayField(TEXT("normals"));
    const auto& UV=Obj->GetArrayField(TEXT("uv"));
    FString Name=FString::Printf(TEXT("SM_Rock_%02d"),Index);
    auto* M=NewObject<UStaticMesh>(CreatePackage(*(Dest+TEXT("Meshes/")+Name)),FName(*Name),RF_Public|RF_Standalone);
    M->GetStaticMaterials().Add(FStaticMaterial(Material));
    FMeshDescription D; FStaticMeshAttributes A(D); A.Register();
    auto Positions=A.GetVertexPositions(); auto Normals=A.GetVertexInstanceNormals();
    auto UVs=A.GetVertexInstanceUVs(); UVs.SetNumChannels(1);
    auto Colors=A.GetVertexInstanceColors();
    for(const auto& V:P) { auto Id=D.CreateVertex(); Positions[Id]=FVector3f(Vec(V->AsArray())); }
    auto Group=D.CreatePolygonGroup();
    for(int j=0;j<I.Num();j+=3)
    {
        TArray<FVertexInstanceID> Corners;
        for(int k=0;k<3;++k)
        {
            int c=j+k; auto VI=D.CreateVertexInstance(FVertexID(int(I[c]->AsNumber())));
            Normals[VI]=FVector3f(Vec(N[c]->AsArray()));
            const auto& U=UV[c]->AsArray(); UVs.Set(VI,0,FVector2f(U[0]->AsNumber(),U[1]->AsNumber()));
            Colors[VI]=FVector4f(1,1,1,1); Corners.Add(VI);
        }
        D.CreatePolygon(Group,Corners);
    }
    auto& Source=M->AddSourceModel(); Source.BuildSettings.bRecomputeNormals=false;
    Source.BuildSettings.bRecomputeTangents=true; Source.BuildSettings.bUseMikkTSpace=true;
    Source.BuildSettings.bUseFullPrecisionUVs=true;
    M->CreateMeshDescription(0,MoveTemp(D)); M->CommitMeshDescription(0);
    M->Build(false); M->CreateBodySetup(); M->GetBodySetup()->CollisionTraceFlag=CTF_UseComplexAsSimple;
    M->GetBodySetup()->CreatePhysicsMeshes();
    FAssetRegistryModule::AssetCreated(M); check(MoonTerrainImport::SaveAsset(M)); return M;
}
int Run()
{
    TMap<FString,UTexture2D*> T;
    for(int i=1;i<=3;i++)
    {
        T.Add(FString::Printf(TEXT("A%d"),i),Texture(FString::Printf(TEXT("T_Soil_%02d_A"),i),true));
        T.Add(FString::Printf(TEXT("N%d"),i),Texture(FString::Printf(TEXT("T_Soil_%02d_N"),i),false));
    }
    T.Add(TEXT("Mask"),Texture(TEXT("T_Mask_01"),false));
    T.Add(TEXT("Detail"),Texture(TEXT("T_Detail_01_N"),false));
    T.Add(TEXT("RockN"),Texture(TEXT("T_Rock_01_N"),false));
    UMaterial* Ground=Material(TEXT("M_MoonLandscape_Full"),false,T);
    UMaterial* Rock=Material(TEXT("M_MoonRock_Full"),true,T);
    UStaticMesh* M1=Mesh(1,Rock); UStaticMesh* M2=Mesh(2,Rock);
    UWorld* W=UEditorLoadingAndSavingUtils::LoadMap(TEXT("/Game/MoonTerrain/Maps/MoonTerrain")); check(W);
    auto Scene=ReadObject(TEXT("scene.json"));
    for(TActorIterator<ALandscape> It(W);It;++It)
    {
        It->SetActorLocation(Vec(Scene->GetArrayField(TEXT("location"))));
        It->SetActorScale3D(Vec(Scene->GetArrayField(TEXT("scale"))));
        It->LandscapeMaterial=Ground; It->UpdateAllComponentMaterialInstances(true);
    }
    FString Text; check(FFileHelper::LoadFileToString(Text,*(Data+TEXT("rocks.json"))));
    TArray<TSharedPtr<FJsonValue>> Records;
    check(FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Records));
    FString CSV=TEXT("actor,id,mesh,x_cm,y_cm,z_cm,qx,qy,qz,qw,sx,sy,sz\n");
    int Count=0;
    for(const auto& V:Records)
    {
        auto O=V->AsObject(); int Id=O->GetIntegerField(TEXT("id"));
        int Kind=O->GetIntegerField(TEXT("mesh"));
        const auto& Rows=O->GetArrayField(TEXT("matrix")); FMatrix Matrix;
        for(int r=0;r<4;r++) for(int c=0;c<4;c++) Matrix.M[r][c]=Rows[r]->AsArray()[c]->AsNumber();
        FTransform Transform(Matrix);
        FActorSpawnParameters P; P.Name=FName(*FString::Printf(TEXT("MoonRock_%06d"),Id));
        AStaticMeshActor* A=W->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(),Transform,P); check(A);
        A->SetActorLabel(P.Name.ToString()); A->SetFolderPath(TEXT("MoonRocks"));
        A->Tags.Add(TEXT("MoonRock")); A->Tags.Add(FName(*FString::Printf(TEXT("USD_Instance_%d"),Id)));
        A->GetStaticMeshComponent()->SetStaticMesh(Kind==1?M1:M2);
        A->GetStaticMeshComponent()->SetCollisionProfileName(TEXT("BlockAll"));
        FVector L=A->GetActorLocation(),S=A->GetActorScale3D(); FQuat Q=A->GetActorQuat();
        CSV+=FString::Printf(TEXT("%s,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n"),*A->GetName(),Id,Kind,L.X,L.Y,L.Z,Q.X,Q.Y,Q.Z,Q.W,S.X,S.Y,S.Z);
        if(++Count%10000==0) UE_LOG(LogMoonTerrainImport,Display,TEXT("ROCKS_CREATED %d"),Count);
    }
    check(FFileHelper::SaveStringToFile(CSV,*(Data+TEXT("rock_positions_ue.csv"))));
    check(UEditorLoadingAndSavingUtils::SaveMap(W,Dest+TEXT("Maps/MoonTerrain_Full")));
    UE_LOG(LogMoonTerrainImport,Display,TEXT("SCENE_V2_SUCCESS rocks=%d map=/Game/MoonTerrainV2/Maps/MoonTerrain_Full"),Count);
    return 0;
}
}
