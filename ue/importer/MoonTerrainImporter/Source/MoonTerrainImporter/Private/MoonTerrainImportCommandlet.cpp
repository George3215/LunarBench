#include "MoonTerrainImportCommandlet.h"
#include "MoonPaths.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/Texture2D.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "Factories/MaterialFactoryNew.h"
#include "Factories/TextureFactory.h"
#include "FileHelpers.h"
#include "HAL/FileManager.h"
#include "Landscape.h"
#include "LandscapeComponent.h"
#include "LandscapeEdit.h"
#include "LandscapeEditTypes.h"
#include "LandscapeHeightfieldCollisionComponent.h"
#include "LandscapeInfo.h"
#include "Materials/Material.h"
#include "Materials/MaterialExpressionComponentMask.h"
#include "Materials/MaterialExpressionConstant.h"
#include "Materials/MaterialExpressionDivide.h"
#include "Materials/MaterialExpressionLinearInterpolate.h"
#include "Materials/MaterialExpressionMultiply.h"
#include "Materials/MaterialExpressionTextureSample.h"
#include "Materials/MaterialExpressionWorldPosition.h"
#include "MaterialEditingLibrary.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "UObject/Package.h"
#include "UObject/SavePackage.h"

DEFINE_LOG_CATEGORY_STATIC(LogMoonTerrainImport, Log, All);

namespace MoonTerrainImport
{
constexpr int32 Resolution = 2795;
constexpr int32 SectionsPerComponent = 1;
constexpr int32 QuadsPerSection = 127;
constexpr double ActorX = 1270.0;
constexpr double ActorY = 2794.0;

const TCHAR* TextureDestination = TEXT("/Game/MoonTerrain/Textures");
const TCHAR* MaterialPackageName = TEXT("/Game/MoonTerrain/Materials/M_MoonLandscape");
const TCHAR* MapAssetPath = TEXT("/Game/MoonTerrain/Maps/MoonTerrain");

bool SaveAsset(UObject* Asset)
{
    if (!Asset)
    {
        return false;
    }

    UPackage* Package = Asset->GetOutermost();
    Package->MarkPackageDirty();
    const FString Filename = FPackageName::LongPackageNameToFilename(
        Package->GetName(), FPackageName::GetAssetPackageExtension());
    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    SaveArgs.SaveFlags = SAVE_NoError;
    return UPackage::SavePackage(Package, Asset, *Filename, SaveArgs);
}

UTexture2D* ImportTexture(
    const FString& SourceFile,
    const FString& AssetName,
    bool bNormal,
    bool bMask)
{
    if (!FPaths::FileExists(SourceFile))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Missing texture: %s"), *SourceFile);
        return nullptr;
    }

    const FString PackageName = FString::Printf(
        TEXT("%s/%s"), TextureDestination, *AssetName);
    const FString ObjectPath = FString::Printf(
        TEXT("%s.%s"), *PackageName, *AssetName);
    UTexture2D* Texture = LoadObject<UTexture2D>(nullptr, *ObjectPath);
    if (!Texture)
    {
        UPackage* Package = CreatePackage(*PackageName);
        UTextureFactory* Factory = NewObject<UTextureFactory>();
        Factory->SuppressImportOverwriteDialog();
        Factory->bUseHashAsGuid = true;
        Factory->LODGroup = bNormal ? TEXTUREGROUP_WorldNormalMap : TEXTUREGROUP_World;
        Factory->ColorSpaceMode = (bNormal || bMask)
            ? ETextureSourceColorSpace::Linear
            : ETextureSourceColorSpace::Auto;
        bool bCancelled = false;
        Texture = Cast<UTexture2D>(Factory->ImportObject(
            UTexture::StaticClass(),
            Package,
            FName(*AssetName),
            RF_Public | RF_Standalone | RF_Transactional,
            SourceFile,
            TEXT(""),
            bCancelled));
        if (Texture)
        {
            FAssetRegistryModule::AssetCreated(Texture);
        }
    }
    if (!Texture)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Texture import failed: %s"), *SourceFile);
        return nullptr;
    }

    Texture->Modify();
    Texture->SRGB = !(bNormal || bMask);
    if (bNormal)
    {
        Texture->CompressionSettings = TC_Normalmap;
        Texture->LODGroup = TEXTUREGROUP_WorldNormalMap;
    }
    else if (bMask)
    {
        Texture->CompressionSettings = TC_Masks;
        Texture->LODGroup = TEXTUREGROUP_World;
    }
    else
    {
        Texture->CompressionSettings = TC_Default;
        Texture->LODGroup = TEXTUREGROUP_World;
    }
    Texture->PostEditChange();
    if (!SaveAsset(Texture))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save texture: %s"), *AssetName);
        return nullptr;
    }
    UE_LOG(LogMoonTerrainImport, Display, TEXT("Imported texture %s -> %s"),
        *SourceFile, *Texture->GetPathName());
    return Texture;
}

template <typename T>
T* CreateExpression(UMaterial* Material, int32 X, int32 Y)
{
    return CastChecked<T>(UMaterialEditingLibrary::CreateMaterialExpression(
        Material, T::StaticClass(), X, Y));
}

UMaterial* CreateLandscapeMaterial(
    UTexture2D* Albedo,
    UTexture2D* Normal,
    UTexture2D* Mask)
{
    UMaterial* Material = LoadObject<UMaterial>(
        nullptr, TEXT("/Game/MoonTerrain/Materials/M_MoonLandscape.M_MoonLandscape"));
    if (!Material)
    {
        UPackage* Package = CreatePackage(MaterialPackageName);
        UMaterialFactoryNew* Factory = NewObject<UMaterialFactoryNew>();
        Material = Cast<UMaterial>(Factory->FactoryCreateNew(
            UMaterial::StaticClass(),
            Package,
            TEXT("M_MoonLandscape"),
            RF_Public | RF_Standalone | RF_Transactional,
            nullptr,
            GWarn));
        if (Material)
        {
            FAssetRegistryModule::AssetCreated(Material);
        }
    }
    if (!Material)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to create landscape material"));
        return nullptr;
    }

    Material->Modify();
    UMaterialEditingLibrary::DeleteAllMaterialExpressions(Material);

    UMaterialExpressionWorldPosition* WorldPosition =
        CreateExpression<UMaterialExpressionWorldPosition>(Material, -1200, 0);
    WorldPosition->WorldPositionShaderOffset = WPT_Default;

    UMaterialExpressionComponentMask* WorldXY =
        CreateExpression<UMaterialExpressionComponentMask>(Material, -1000, 0);
    WorldXY->R = true;
    WorldXY->G = true;
    WorldXY->B = false;
    WorldXY->A = false;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldPosition, TEXT(""), WorldXY, TEXT("Input"));

    // USD material override: Soil_01_NearTiling = 400 cm.
    UMaterialExpressionDivide* SoilUV =
        CreateExpression<UMaterialExpressionDivide>(Material, -800, -140);
    SoilUV->ConstB = 400.0f;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldXY, TEXT(""), SoilUV, TEXT("A"));

    UMaterialExpressionTextureSample* AlbedoSample =
        CreateExpression<UMaterialExpressionTextureSample>(Material, -560, -240);
    AlbedoSample->Texture = Albedo;
    AlbedoSample->SamplerType = SAMPLERTYPE_Color;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        SoilUV, TEXT(""), AlbedoSample, TEXT("Coordinates"));

    UMaterialExpressionTextureSample* NormalSample =
        CreateExpression<UMaterialExpressionTextureSample>(Material, -560, 220);
    NormalSample->Texture = Normal;
    NormalSample->SamplerType = SAMPLERTYPE_Normal;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        SoilUV, TEXT(""), NormalSample, TEXT("Coordinates"));

    // USD override: the broad colorizer mask uses a 16384 cm world scale and
    // darkens masked regions to 51% of their base color.
    UMaterialExpressionDivide* MaskUV =
        CreateExpression<UMaterialExpressionDivide>(Material, -800, 520);
    MaskUV->ConstB = 16384.0f;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldXY, TEXT(""), MaskUV, TEXT("A"));

    UMaterialExpressionTextureSample* MaskSample =
        CreateExpression<UMaterialExpressionTextureSample>(Material, -560, 520);
    MaskSample->Texture = Mask;
    MaskSample->SamplerType = SAMPLERTYPE_Masks;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        MaskUV, TEXT(""), MaskSample, TEXT("Coordinates"));

    UMaterialExpressionLinearInterpolate* Colorizer =
        CreateExpression<UMaterialExpressionLinearInterpolate>(Material, -300, -20);
    Colorizer->ConstA = 1.0f;
    Colorizer->ConstB = 0.51f;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        MaskSample, TEXT("A"), Colorizer, TEXT("Alpha"));

    UMaterialExpressionMultiply* TintedAlbedo =
        CreateExpression<UMaterialExpressionMultiply>(Material, -80, -160);
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        AlbedoSample, TEXT("RGB"), TintedAlbedo, TEXT("A"));
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        Colorizer, TEXT(""), TintedAlbedo, TEXT("B"));

    UMaterialExpressionConstant* Roughness =
        CreateExpression<UMaterialExpressionConstant>(Material, -80, 300);
    Roughness->R = 1.0f;
    UMaterialExpressionConstant* Specular =
        CreateExpression<UMaterialExpressionConstant>(Material, -80, 400);
    Specular->R = 0.25f;

    if (!UMaterialEditingLibrary::ConnectMaterialProperty(
            TintedAlbedo, TEXT(""), MP_BaseColor) ||
        !UMaterialEditingLibrary::ConnectMaterialProperty(
            NormalSample, TEXT("RGB"), MP_Normal) ||
        !UMaterialEditingLibrary::ConnectMaterialProperty(
            Roughness, TEXT(""), MP_Roughness) ||
        !UMaterialEditingLibrary::ConnectMaterialProperty(
            Specular, TEXT(""), MP_Specular))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to connect material graph"));
        return nullptr;
    }

    // UE 5.7 removed the old MATUSAGE_Landscape flag. Landscape materials are
    // identified by the landscape rendering path when they are assigned.
    UMaterialEditingLibrary::RecompileMaterial(Material);
    Material->PostEditChange();
    if (!SaveAsset(Material))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save material"));
        return nullptr;
    }
    UE_LOG(LogMoonTerrainImport, Display, TEXT("Created material %s"),
        *Material->GetPathName());
    return Material;
}

bool LoadHeightData(const FString& Filename, TArray<uint16>& OutHeights)
{
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *Filename))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Could not read R16 file: %s"), *Filename);
        return false;
    }
    const int64 ExpectedBytes = static_cast<int64>(Resolution) * Resolution * sizeof(uint16);
    if (Bytes.Num() != ExpectedBytes)
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("R16 byte count mismatch: got %d, expected %lld"),
            Bytes.Num(), ExpectedBytes);
        return false;
    }
#if !PLATFORM_LITTLE_ENDIAN
    UE_LOG(LogMoonTerrainImport, Error, TEXT("The prepared R16 file is little-endian"));
    return false;
#else
    OutHeights.SetNumUninitialized(Resolution * Resolution);
    FMemory::Memcpy(OutHeights.GetData(), Bytes.GetData(), Bytes.Num());
    return true;
#endif
}

bool ValidateImportedAssets(const FString& Heightmap)
{
    const FString MapFilename = FPackageName::LongPackageNameToFilename(
        MapAssetPath, FPackageName::GetMapPackageExtension());
    UWorld* World = UEditorLoadingAndSavingUtils::LoadMap(MapFilename);
    if (!World)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("VALIDATION: failed to reload map %s"),
            *MapFilename);
        return false;
    }

    ALandscape* Landscape = nullptr;
    int32 LandscapeActorCount = 0;
    for (TActorIterator<ALandscape> It(World); It; ++It)
    {
        Landscape = *It;
        ++LandscapeActorCount;
    }
    if (!Landscape || LandscapeActorCount != 1)
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("VALIDATION: expected exactly one ALandscape, found %d"),
            LandscapeActorCount);
        return false;
    }

    TArray<ULandscapeComponent*> Components;
    Landscape->GetComponents(Components);
    TArray<ULandscapeHeightfieldCollisionComponent*> CollisionComponents;
    Landscape->GetComponents(CollisionComponents);

    int32 MinSectionX = MAX_int32;
    int32 MinSectionY = MAX_int32;
    int32 MaxVertexX = MIN_int32;
    int32 MaxVertexY = MIN_int32;
    bool bComponentLayoutValid = Components.Num() == 22 * 22;
    for (const ULandscapeComponent* Component : Components)
    {
        if (!Component)
        {
            bComponentLayoutValid = false;
            continue;
        }
        MinSectionX = FMath::Min(MinSectionX, Component->SectionBaseX);
        MinSectionY = FMath::Min(MinSectionY, Component->SectionBaseY);
        MaxVertexX = FMath::Max(
            MaxVertexX, Component->SectionBaseX + Component->ComponentSizeQuads);
        MaxVertexY = FMath::Max(
            MaxVertexY, Component->SectionBaseY + Component->ComponentSizeQuads);
        bComponentLayoutValid &= Component->ComponentSizeQuads == QuadsPerSection;
        bComponentLayoutValid &= Component->NumSubsections == SectionsPerComponent;
        bComponentLayoutValid &= Component->SubsectionSizeQuads == QuadsPerSection;
    }
    const int32 LoadedResolutionX = MaxVertexX - MinSectionX + 1;
    const int32 LoadedResolutionY = MaxVertexY - MinSectionY + 1;
    bComponentLayoutValid &= MinSectionX == 0 && MinSectionY == 0;
    bComponentLayoutValid &= LoadedResolutionX == Resolution;
    bComponentLayoutValid &= LoadedResolutionY == Resolution;

    const FVector Location = Landscape->GetActorLocation();
    const FVector Scale = Landscape->GetActorScale3D();
    const bool bTransformValid = Location.Equals(FVector(ActorX, ActorY, 0.0), 0.001)
        && Scale.Equals(FVector(1.0, 1.0, 1.0), 0.000001);

    UMaterialInterface* AssignedMaterial = Landscape->GetLandscapeMaterial();
    const FString ExpectedMaterialPath =
        TEXT("/Game/MoonTerrain/Materials/M_MoonLandscape.M_MoonLandscape");
    const bool bMaterialValid = AssignedMaterial
        && AssignedMaterial->GetPathName() == ExpectedMaterialPath;

    UTexture2D* Albedo = LoadObject<UTexture2D>(
        nullptr, TEXT("/Game/MoonTerrain/Textures/T_Soil_01_A.T_Soil_01_A"));
    UTexture2D* Normal = LoadObject<UTexture2D>(
        nullptr, TEXT("/Game/MoonTerrain/Textures/T_Soil_01_N.T_Soil_01_N"));
    UTexture2D* Mask = LoadObject<UTexture2D>(
        nullptr, TEXT("/Game/MoonTerrain/Textures/T_Mask_01.T_Mask_01"));
    const FIntPoint AlbedoImportedSize = Albedo ? Albedo->GetImportedSize() : FIntPoint::ZeroValue;
    const FIntPoint NormalImportedSize = Normal ? Normal->GetImportedSize() : FIntPoint::ZeroValue;
    const FIntPoint MaskImportedSize = Mask ? Mask->GetImportedSize() : FIntPoint::ZeroValue;
    const bool bAlbedoValid = Albedo && AlbedoImportedSize == FIntPoint(8192, 8192)
        && Albedo->SRGB
        && Albedo->CompressionSettings == TC_Default;
    const bool bNormalValid = Normal && NormalImportedSize == FIntPoint(8192, 8192)
        && !Normal->SRGB
        && Normal->CompressionSettings == TC_Normalmap;
    const bool bMaskValid = Mask && MaskImportedSize == FIntPoint(4096, 4096)
        && !Mask->SRGB
        && Mask->CompressionSettings == TC_Masks;

    bool bReferencesAlbedo = false;
    bool bReferencesNormal = false;
    bool bReferencesMask = false;
    if (AssignedMaterial)
    {
        for (UObject* ReferencedObject : AssignedMaterial->GetReferencedTextures())
        {
            bReferencesAlbedo |= ReferencedObject == Albedo;
            bReferencesNormal |= ReferencedObject == Normal;
            bReferencesMask |= ReferencedObject == Mask;
        }
    }
    const bool bTextureReferencesValid =
        bReferencesAlbedo && bReferencesNormal && bReferencesMask;

    TArray<uint16> ExpectedHeights;
    bool bHeightDataValid = LoadHeightData(Heightmap, ExpectedHeights);
    int64 HeightMismatchCount = -1;
    uint16 LoadedMinHeight = MAX_uint16;
    uint16 LoadedMaxHeight = 0;
    if (bHeightDataValid)
    {
        ULandscapeInfo* Info = Landscape->GetLandscapeInfo();
        if (!Info)
        {
            bHeightDataValid = false;
        }
        else
        {
            TArray<uint16> LoadedHeights;
            LoadedHeights.SetNumUninitialized(Resolution * Resolution);
            FLandscapeEditDataInterface Edit(Info, false);
            Edit.GetHeightDataFast(
                0, 0, Resolution - 1, Resolution - 1,
                LoadedHeights.GetData(), Resolution);
            HeightMismatchCount = 0;
            for (int32 Index = 0; Index < LoadedHeights.Num(); ++Index)
            {
                LoadedMinHeight = FMath::Min(LoadedMinHeight, LoadedHeights[Index]);
                LoadedMaxHeight = FMath::Max(LoadedMaxHeight, LoadedHeights[Index]);
                if (LoadedHeights[Index] != ExpectedHeights[Index])
                {
                    ++HeightMismatchCount;
                }
            }
            bHeightDataValid = HeightMismatchCount == 0;
        }
    }

    const bool bCollisionValid = CollisionComponents.Num() == Components.Num();
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_GEOMETRY: actors=%d components=%d collision=%d resolution=%dx%d section_base=(%d,%d)-(%d,%d)"),
        LandscapeActorCount, Components.Num(), CollisionComponents.Num(),
        LoadedResolutionX, LoadedResolutionY, MinSectionX, MinSectionY,
        MaxVertexX, MaxVertexY);
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_TRANSFORM: location_cm=(%.6f,%.6f,%.6f) scale=(%.6f,%.6f,%.6f)"),
        Location.X, Location.Y, Location.Z, Scale.X, Scale.Y, Scale.Z);
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_HEIGHTS: samples=%d mismatches=%lld encoded_min=%u encoded_max=%u z_cm=[%.9f,%.9f]"),
        ExpectedHeights.Num(), HeightMismatchCount, LoadedMinHeight, LoadedMaxHeight,
        (static_cast<double>(LoadedMinHeight) - 32768.0) / 128.0,
        (static_cast<double>(LoadedMaxHeight) - 32768.0) / 128.0);
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_MATERIAL: assigned=%s refs=(albedo:%s normal:%s mask:%s)"),
        AssignedMaterial ? *AssignedMaterial->GetPathName() : TEXT("None"),
        bReferencesAlbedo ? TEXT("yes") : TEXT("no"),
        bReferencesNormal ? TEXT("yes") : TEXT("no"),
        bReferencesMask ? TEXT("yes") : TEXT("no"));
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_TEXTURES: albedo=%dx%d sRGB=%s compression=%d normal=%dx%d sRGB=%s compression=%d mask=%dx%d sRGB=%s compression=%d"),
        AlbedoImportedSize.X, AlbedoImportedSize.Y,
        Albedo && Albedo->SRGB ? TEXT("true") : TEXT("false"),
        Albedo ? static_cast<int32>(Albedo->CompressionSettings) : -1,
        NormalImportedSize.X, NormalImportedSize.Y,
        Normal && Normal->SRGB ? TEXT("true") : TEXT("false"),
        Normal ? static_cast<int32>(Normal->CompressionSettings) : -1,
        MaskImportedSize.X, MaskImportedSize.Y,
        Mask && Mask->SRGB ? TEXT("true") : TEXT("false"),
        Mask ? static_cast<int32>(Mask->CompressionSettings) : -1);

    const bool bAllValid = bComponentLayoutValid && bCollisionValid
        && bTransformValid && bMaterialValid && bAlbedoValid && bNormalValid
        && bMaskValid && bTextureReferencesValid && bHeightDataValid;
    if (bAllValid)
    {
        UE_LOG(LogMoonTerrainImport, Display, TEXT("VALIDATION_RESULT: PASS"));
    }
    else
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("VALIDATION_RESULT: FAIL"));
    }
    return bAllValid;
}

ALandscape* CreateLandscape(UWorld* World, TArray<uint16>&& Heights, UMaterial* Material)
{
    FActorSpawnParameters SpawnParameters;
    SpawnParameters.Name = TEXT("MoonLandscape");
    ALandscape* Landscape = World->SpawnActor<ALandscape>(
        FVector(ActorX, ActorY, 0.0), FRotator::ZeroRotator, SpawnParameters);
    if (!Landscape)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to spawn ALandscape"));
        return nullptr;
    }
    Landscape->SetActorScale3D(FVector(1.0, 1.0, 1.0));
    // EditorSetLandscapeMaterial is Blueprint-visible but is not exported by
    // the UE 5.7 Landscape binary on Linux.  The property itself is public;
    // UpdateAllComponentMaterialInstances below performs the required refresh.
    Landscape->LandscapeMaterial = Material;

    TMap<FGuid, TArray<uint16>> HeightDataPerLayer;
    HeightDataPerLayer.Add(FGuid(), MoveTemp(Heights));
    TMap<FGuid, TArray<FLandscapeImportLayerInfo>> MaterialLayerData;
    MaterialLayerData.Add(FGuid(), TArray<FLandscapeImportLayerInfo>());
    Landscape->Import(
        FGuid::NewGuid(),
        0, 0, Resolution - 1, Resolution - 1,
        SectionsPerComponent, QuadsPerSection,
        HeightDataPerLayer,
        nullptr,
        MaterialLayerData,
        ELandscapeImportAlphamapType::Additive,
        TArrayView<const FLandscapeLayer>());
    Landscape->SetActorLabel(TEXT("Moon Landscape (Landscape_1.usd)"));
    Landscape->ReimportHeightmapFilePath = TEXT("");
    Landscape->StaticLightingLOD = 1;
    Landscape->CreateLandscapeInfo();
    Landscape->UpdateAllComponentMaterialInstances(true);
    Landscape->MarkPackageDirty();
    return Landscape;
}

void AddLighting(UWorld* World)
{
    FActorSpawnParameters SunParams;
    SunParams.Name = TEXT("MoonSun");
    ADirectionalLight* Sun = World->SpawnActor<ADirectionalLight>(
        FVector::ZeroVector, FRotator(-35.0, -135.0, 0.0), SunParams);
    if (Sun && Sun->GetComponent())
    {
        Sun->SetActorLabel(TEXT("Moon Sun"));
        Sun->GetComponent()->SetIntensity(7.0f);
        Sun->GetComponent()->SetLightColor(FLinearColor(1.0f, 0.98f, 0.92f));
    }

    FActorSpawnParameters SkyParams;
    SkyParams.Name = TEXT("MoonSkyLight");
    ASkyLight* Sky = World->SpawnActor<ASkyLight>(
        FVector::ZeroVector, FRotator::ZeroRotator, SkyParams);
    if (Sky && Sky->GetLightComponent())
    {
        Sky->SetActorLabel(TEXT("Moon Fill Light"));
        Sky->GetLightComponent()->SetIntensity(0.12f);
        Sky->GetLightComponent()->Mobility = EComponentMobility::Movable;
    }
}
} // namespace MoonTerrainImport

#include "MoonSceneV2.inl"
#include "MoonGo2.inl"
#include "MoonMacro.inl"

UMoonTerrainImportCommandlet::UMoonTerrainImportCommandlet()
{
    IsClient = false;
    IsServer = false;
    IsEditor = true;
    LogToConsole = true;
    ShowErrorCount = true;
}

int32 UMoonTerrainImportCommandlet::Main(const FString& Params)
{
    using namespace MoonTerrainImport;

    if (FParse::Param(*Params, TEXT("UpgradeScene"))) return MoonSceneV2::Run();
    if (FParse::Param(*Params, TEXT("ImportGo2"))) return MoonGo2::Import();

    // 只换地形材质、另存为新地图；源地图与源材质只读。
    if (FParse::Param(*Params, TEXT("MoonMacroMap"))) return MoonMacro::Run(Params);
    if (FParse::Param(*Params, TEXT("ValidateMacroMap")))
    {
        return MoonMacro::Validate(Params) ? 0 : 11;
    }

    if (FParse::Param(*Params, TEXT("ValidateOnly")))
    {
        FString ValidationHeightmap;
        if (!FParse::Value(*Params, TEXT("Heightmap="), ValidationHeightmap)
            || ValidationHeightmap.IsEmpty())
        {
            UE_LOG(LogMoonTerrainImport, Error,
                TEXT("Validation requires -Heightmap=/absolute/path/to/Landscape_1_2795x2795.r16"));
            return 10;
        }
        ValidationHeightmap = FPaths::ConvertRelativePathToFull(ValidationHeightmap);
        return ValidateImportedAssets(ValidationHeightmap) ? 0 : 11;
    }

    FString SourceRoot;
    FString Heightmap;
    if (!FParse::Value(*Params, TEXT("SourceRoot="), SourceRoot) || SourceRoot.IsEmpty())
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("Required argument: -SourceRoot=/absolute/path/to/landscape_cropped"));
        return 2;
    }
    if (!FParse::Value(*Params, TEXT("Heightmap="), Heightmap) || Heightmap.IsEmpty())
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("Required argument: -Heightmap=/absolute/path/to/Landscape_1_2795x2795.r16"));
        return 2;
    }
    SourceRoot = FPaths::ConvertRelativePathToFull(SourceRoot);
    Heightmap = FPaths::ConvertRelativePathToFull(Heightmap);

    const FString TextureRoot = FPaths::Combine(SourceRoot, TEXT("Materials/Textures"));
    const FString AlbedoFile = FPaths::Combine(TextureRoot, TEXT("T_Soil_01_A.png"));
    const FString NormalFile = FPaths::Combine(TextureRoot, TEXT("T_Soil_01_N.png"));
    const FString MaskFile = FPaths::Combine(TextureRoot, TEXT("T_Mask_01.png"));
    const FString SourceUsd = FPaths::Combine(SourceRoot, TEXT("Props/Landscape_1.usd"));

    if (!FPaths::FileExists(SourceUsd) || !FPaths::FileExists(Heightmap))
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("Missing source USD or prepared heightmap. USD=%s Heightmap=%s"),
            *SourceUsd, *Heightmap);
        return 3;
    }

    UTexture2D* Albedo = ImportTexture(
        AlbedoFile, TEXT("T_Soil_01_A"), false, false);
    UTexture2D* Normal = ImportTexture(
        NormalFile, TEXT("T_Soil_01_N"), true, false);
    UTexture2D* Mask = ImportTexture(
        MaskFile, TEXT("T_Mask_01"), false, true);
    if (!Albedo || !Normal || !Mask)
    {
        return 4;
    }

    UMaterial* Material = CreateLandscapeMaterial(Albedo, Normal, Mask);
    if (!Material)
    {
        return 5;
    }

    TArray<uint16> Heights;
    if (!LoadHeightData(Heightmap, Heights))
    {
        return 6;
    }

    UWorld* World = UEditorLoadingAndSavingUtils::NewBlankMap(false);
    if (!World)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to create blank editor map"));
        return 7;
    }
    ALandscape* Landscape = CreateLandscape(World, MoveTemp(Heights), Material);
    if (!Landscape)
    {
        return 8;
    }
    AddLighting(World);

    if (!UEditorLoadingAndSavingUtils::SaveMap(World, MapAssetPath))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save map %s"), MapAssetPath);
        return 9;
    }

    FAssetRegistryModule::AssetCreated(Landscape);
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("SUCCESS: native Landscape imported from %s"), *SourceUsd);
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("Map=%s Material=%s Components=22x22 Resolution=2795x2795"),
        MapAssetPath, MaterialPackageName);
    return 0;
}
