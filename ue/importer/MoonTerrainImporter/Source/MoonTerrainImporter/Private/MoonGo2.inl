#include "Materials/MaterialExpressionConstant3Vector.h"
namespace MoonGo2
{
int Import()
{
    FString Text;
    check(FFileHelper::LoadFileToString(Text,*MoonPaths::File(TEXT("mujoco/generated/visual.json"))));
    TSharedPtr<FJsonObject> Root;
    check(FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Root));
    TMap<int,UStaticMesh*> Meshes;
    for(const auto& Value:Root->GetArrayField(TEXT("meshes")))
    {
        auto O=Value->AsObject();int Id=O->GetIntegerField(TEXT("id"));
        FString Name=FString::Printf(TEXT("SM_Go2_%d"),Id);
        if(auto* Existing=LoadObject<UStaticMesh>(nullptr,*(TEXT("/Game/MoonGo2/Meshes/")+Name))) {Meshes.Add(Id,Existing);continue;}
        auto* M=NewObject<UStaticMesh>(CreatePackage(*(TEXT("/Game/MoonGo2/Meshes/")+Name)),FName(*Name),RF_Public|RF_Standalone);
        M->GetStaticMaterials().Add(FStaticMaterial());
        FMeshDescription D;FStaticMeshAttributes A(D);A.Register();
        auto Positions=A.GetVertexPositions();auto UV=A.GetVertexInstanceUVs();UV.SetNumChannels(1);
        for(const auto& V:O->GetArrayField(TEXT("points")))
        {auto VId=D.CreateVertex();Positions[VId]=FVector3f(MoonSceneV2::Vec(V->AsArray()));}
        const auto& Indices=O->GetArrayField(TEXT("indices"));auto Group=D.CreatePolygonGroup();
        for(int i=0;i<Indices.Num();i+=3)
        {
            TArray<FVertexInstanceID> Corners;
            for(int k=0;k<3;k++){auto VI=D.CreateVertexInstance(FVertexID(int(Indices[i+k]->AsNumber())));UV.Set(VI,0,FVector2f::ZeroVector);Corners.Add(VI);}
            D.CreatePolygon(Group,Corners);
        }
        auto& Source=M->AddSourceModel();Source.BuildSettings.bRecomputeNormals=true;Source.BuildSettings.bRecomputeTangents=true;
        M->CreateMeshDescription(0,MoveTemp(D));M->CommitMeshDescription(0);M->Build(false);
        FAssetRegistryModule::AssetCreated(M);check(MoonTerrainImport::SaveAsset(M));Meshes.Add(Id,M);
    }
    // Build a genuinely local world. Never load the full Landscape or full rock map.
    FString PatchText;check(FFileHelper::LoadFileToString(PatchText,*MoonPaths::File(TEXT("mujoco/generated/patch.json"))));
    TSharedPtr<FJsonObject> Patch;check(FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(PatchText),Patch));
    const bool Full=FParse::Param(FCommandLine::Get(),TEXT("FullGo2Render"));
    UWorld* W=Full ? UEditorLoadingAndSavingUtils::LoadMap(TEXT("/Game/MoonTerrainV2/Maps/MoonTerrain_Full")) : UEditorLoadingAndSavingUtils::NewBlankMap(false);check(W);
    if(!Full)
    {
    MoonTerrainImport::AddLighting(W);
    FVector Origin=MoonSceneV2::Vec(Patch->GetArrayField(TEXT("origin_cm")));
    UStaticMesh* Ground=LoadObject<UStaticMesh>(nullptr,TEXT("/Game/MoonGo2/Meshes/SM_LocalTerrain10m_Matched"));
    if(!Ground)
    {
        Ground=NewObject<UStaticMesh>(CreatePackage(TEXT("/Game/MoonGo2/Meshes/SM_LocalTerrain10m_Matched")),TEXT("SM_LocalTerrain10m_Matched"),RF_Public|RF_Standalone);
        Ground->GetStaticMaterials().Add(FStaticMaterial(LoadObject<UMaterial>(nullptr,TEXT("/Game/MoonTerrainV2/Materials/M_MoonLandscape_Full"))));
        FMeshDescription D;FStaticMeshAttributes A(D);A.Register();auto P=A.GetVertexPositions();auto UV=A.GetVertexInstanceUVs();UV.SetNumChannels(1);auto Normals=A.GetVertexInstanceNormals();
        for(const auto& V:Patch->GetArrayField(TEXT("points"))){auto Id=D.CreateVertex();P[Id]=FVector3f(MoonSceneV2::Vec(V->AsArray()));}
        const auto& Indices=Patch->GetArrayField(TEXT("indices"));auto Group=D.CreatePolygonGroup();
        for(int i=0;i<Indices.Num();i+=3)
        {
            TArray<FVertexInstanceID> Corners;
            for(int k=0;k<3;k++){auto Vertex=FVertexID(int(Indices[i+k]->AsNumber()));auto VI=D.CreateVertexInstance(Vertex);UV.Set(VI,0,FVector2f(P[Vertex].X/50,P[Vertex].Y/50));Normals[VI]=FVector3f(MoonSceneV2::Vec(Patch->GetArrayField(TEXT("normals"))[Vertex.GetValue()]->AsArray()));Corners.Add(VI);}
            D.CreatePolygon(Group,Corners);
        }
        auto& S=Ground->AddSourceModel();S.BuildSettings.bRecomputeNormals=false;S.BuildSettings.bRecomputeTangents=true;S.BuildSettings.bUseFullPrecisionUVs=true;
        Ground->CreateMeshDescription(0,MoveTemp(D));Ground->CommitMeshDescription(0);Ground->Build(false);FAssetRegistryModule::AssetCreated(Ground);check(MoonTerrainImport::SaveAsset(Ground));
    }
    auto* G=W->SpawnActor<AStaticMeshActor>(Origin,FRotator::ZeroRotator);G->SetActorLabel(TEXT("LocalTerrain_10m_441samples"));G->Tags.Add(TEXT("LocalTerrain"));G->GetStaticMeshComponent()->SetStaticMesh(Ground);G->SetActorEnableCollision(false);
    for(const auto& V:Patch->GetArrayField(TEXT("rocks")))
    {
        auto O=V->AsObject();FActorSpawnParameters P;P.Name=FName(*FString::Printf(TEXT("MoonRock_%06d"),O->GetIntegerField(TEXT("id"))));
        const auto& Q=O->GetArrayField(TEXT("quaternion"));FQuat R(Q[0]->AsNumber(),Q[1]->AsNumber(),Q[2]->AsNumber(),Q[3]->AsNumber());
        auto* A=W->SpawnActor<AStaticMeshActor>(MoonSceneV2::Vec(O->GetArrayField(TEXT("position"))),R.Rotator(),P);
        A->SetActorScale3D(MoonSceneV2::Vec(O->GetArrayField(TEXT("scale"))));A->SetActorLabel(P.Name.ToString());A->Tags.Add(TEXT("MoonRock"));A->SetActorEnableCollision(false);
        A->GetStaticMeshComponent()->SetStaticMesh(LoadObject<UStaticMesh>(nullptr,*FString::Printf(TEXT("/Game/MoonTerrainV2/Meshes/SM_Rock_%02d"),O->GetIntegerField(TEXT("mesh")))));
    }
    }
    // UE is rendering-only, including all Landscape collision components.
    for(TActorIterator<AActor> It(W);It;++It)
    {
        It->SetActorEnableCollision(false);
        // Landscape collision reads its owner's BodyInstance, not the component's.
        if(auto* Landscape=Cast<ALandscapeProxy>(*It))
        {
            Landscape->BodyInstance.SetCollisionProfileName(TEXT("NoCollision"));
            Landscape->BodyInstance.SetCollisionEnabled(ECollisionEnabled::NoCollision);
        }
        TInlineComponentArray<UPrimitiveComponent*> Components;It->GetComponents(Components);
        for(auto* C:Components){C->SetSimulatePhysics(false);C->SetCollisionProfileName(TEXT("NoCollision"));C->SetCollisionEnabled(ECollisionEnabled::NoCollision);}
    }
    TMap<int,TArray<TSharedPtr<FJsonValue>>> Poses;
    for(const auto& P:Root->GetObjectField(TEXT("initial"))->GetArrayField(TEXT("geoms")))
    {const auto& A=P->AsArray();Poses.Add(int(A[0]->AsNumber()),A);}
    for(const auto& Value:Root->GetArrayField(TEXT("geoms")))
    {
        auto O=Value->AsObject();int Id=O->GetIntegerField(TEXT("id"));
        FString Name=FString::Printf(TEXT("Go2Geom_%03d"),Id);
        const auto& P=Poses[Id];FVector Location(P[1]->AsNumber(),P[2]->AsNumber(),P[3]->AsNumber());
        FQuat Q(P[4]->AsNumber(),P[5]->AsNumber(),P[6]->AsNumber(),P[7]->AsNumber());
        FActorSpawnParameters Params;Params.Name=FName(*Name);
        auto* Actor=W->SpawnActor<AStaticMeshActor>(Location,Q.Rotator(),Params);
        Actor->SetActorLabel(Name);Actor->SetFolderPath(TEXT("Go2_MuJoCo"));Actor->Tags.Add(TEXT("Go2Visual"));Actor->SetActorEnableCollision(false);
        auto* C=Actor->GetStaticMeshComponent();C->SetMobility(EComponentMobility::Movable);
        C->SetStaticMesh(Meshes[O->GetIntegerField(TEXT("mesh"))]);C->SetSimulatePhysics(false);C->SetCollisionProfileName(TEXT("NoCollision"));C->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        auto* Factory=NewObject<UMaterialFactoryNew>();FString MatName=TEXT("M_")+Name;
        if(auto* Existing=LoadObject<UMaterial>(nullptr,*(TEXT("/Game/MoonGo2/Materials/")+MatName))) {C->SetMaterial(0,Existing);continue;}
        auto* Mat=CastChecked<UMaterial>(Factory->FactoryCreateNew(UMaterial::StaticClass(),CreatePackage(*(TEXT("/Game/MoonGo2/Materials/")+MatName)),FName(*MatName),RF_Public|RF_Standalone,nullptr,GWarn));
        auto* Color=MoonTerrainImport::CreateExpression<UMaterialExpressionConstant3Vector>(Mat,0,0);
        const auto& RGBA=O->GetArrayField(TEXT("rgba"));Color->Constant=FLinearColor(RGBA[0]->AsNumber(),RGBA[1]->AsNumber(),RGBA[2]->AsNumber());
        UMaterialEditingLibrary::ConnectMaterialProperty(Color,TEXT(""),MP_BaseColor);
        UMaterialEditingLibrary::RecompileMaterial(Mat);Mat->PostEditChange();FAssetRegistryModule::AssetCreated(Mat);
        check(MoonTerrainImport::SaveAsset(Mat));C->SetMaterial(0,Mat);
    }
    check(UEditorLoadingAndSavingUtils::SaveMap(W,Full ? TEXT("/Game/MoonGo2/Maps/MoonTerrain_Go2_FullRender") : TEXT("/Game/MoonGo2/Maps/MoonTerrain_Go2_Patch")));
    UE_LOG(LogMoonTerrainImport,Display,TEXT("GO2_IMPORT_PASS meshes=%d"),Meshes.Num());return 0;
}
}
