// 用一套 PBR 贴图给地形换皮，另存为一张新地图。
//
// 与 MoonSceneV2 / MoonGo2 的关系：那两者负责「造」地图（导入高度场与岩石），
// 本文件只负责「换皮」——载入一张已有地图，改掉其中 ALandscape 的材质，另存到新路径。
// 源地图与源材质一律只读，绝不回写。
//
// 几何完全不动：不改高度数据、不改 transform、不接收任何 displacement 贴图。
// 仓库的核心不变量是 UE 渲染的必须是 MuJoCo 那份 1 cm 原始高度场。
#include "Materials/MaterialExpressionNormalize.h"
#include "Materials/MaterialExpressionScalarParameter.h"
#include "Materials/MaterialExpressionVectorParameter.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Materials/MaterialInterface.h"
#include "Factories/MaterialInstanceConstantFactoryNew.h"

namespace MoonMacro
{
// 本文件被 include 在 `} // namespace MoonTerrainImport` 之后（与 MoonSceneV2 / MoonGo2 同处），
// 所以 SaveAsset / CreateExpression / LoadHeightData 与那几个常量需要显式引入。
using namespace MoonTerrainImport;

const TCHAR* SrcMapDefault = TEXT("/Game/MoonGo2/Maps/MoonTerrain_Go2_FullRender");
const TCHAR* DstMapDefault = TEXT("/Game/MoonMacro/Maps/MoonTerrain_Macro01_FullRender");
const TCHAR* TextureDestDefault = TEXT("/Game/MoonMacro/Textures");
const TCHAR* MaterialDefault = TEXT("/Game/MoonMacro/Materials/M_MoonMacro_Landscape");
const TCHAR* InstanceDefault = TEXT("/Game/MoonMacro/Materials/MI_MoonMacro01_Landscape");
const TCHAR* TextureRootDefault = TEXT("assets/environments/lunar/materials/moon_macro_01_4k");

// 三张贴图的角色决定它该用什么颜色空间与压缩格式，这三者必须成套改，不能只改一半。
enum class Channel
{
    Albedo,
    Normal,
    Roughness,
};

struct Options
{
    FString SrcMap = SrcMapDefault;
    FString DstMap = DstMapDefault;
    FString TextureDest = TextureDestDefault;
    FString Material = MaterialDefault;
    FString Instance = InstanceDefault;
    FString TextureRoot = TextureRootDefault;
    double TileCm = 1000.0;      // 世界空间平铺边长，见下方 UV 注释
    double Specular = 0.35;
    double NormalScale = 1.0;    // 1.0 时 Multiply+Normalize 是恒等变换
};

// 把一张贴图导进 UE，并强制设定它作为该通道应有的采样方式。
// 已存在也照样重设属性：上一次跑到一半失败会留下设置不全的资产，不能信任。
UTexture2D* ImportTexture(
    const FString& SourceDirectory,
    const FString& FileName,
    const FString& AssetName,
    const FString& Destination,
    Channel Kind)
{
    const FString Source = FPaths::Combine(SourceDirectory, FileName);
    if (!FPaths::FileExists(Source))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Missing source texture: %s"), *Source);
        return nullptr;
    }

    const FString PackageName = FString::Printf(TEXT("%s/%s"), *Destination, *AssetName);
    const FString ObjectPath = FString::Printf(TEXT("%s.%s"), *PackageName, *AssetName);
    UTexture2D* Texture = LoadObject<UTexture2D>(nullptr, *ObjectPath);
    if (!Texture)
    {
        UPackage* Package = CreatePackage(*PackageName);
        UTextureFactory* Factory = NewObject<UTextureFactory>();
        Factory->SuppressImportOverwriteDialog();
        // 法线必须按线性数据读入；当成 sRGB 读会让法线方向整体偏掉。
        Factory->ColorSpaceMode = Kind == Channel::Albedo
            ? ETextureSourceColorSpace::Auto
            : ETextureSourceColorSpace::Linear;
        bool bCancelled = false;
        Texture = Cast<UTexture2D>(Factory->ImportObject(
            UTexture::StaticClass(),
            Package,
            FName(*AssetName),
            RF_Public | RF_Standalone | RF_Transactional,
            Source,
            TEXT(""),
            bCancelled));
        if (Texture)
        {
            FAssetRegistryModule::AssetCreated(Texture);
        }
    }
    if (!Texture)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Texture import failed: %s"), *Source);
        return nullptr;
    }

    Texture->Modify();
    switch (Kind)
    {
    case Channel::Albedo:
        Texture->SRGB = true;
        // 不用 MoonSceneV2 那套 BC7：那里是为了保住打包格式的 B/A 通道
        // （源 Soil_N 的 B 是 specular、A 是混合高度）。本贴图的基色无打包，BC7 即可。
        Texture->CompressionSettings = TC_BC7;
        Texture->LODGroup = TEXTUREGROUP_Cinematic;
        Texture->bFlipGreenChannel = false;
        break;
    case Channel::Normal:
        Texture->SRGB = false;
        // 纯法线图，BC5 质量更高且内存减半。
        Texture->CompressionSettings = TC_Normalmap;
        Texture->LODGroup = TEXTUREGROUP_WorldNormalMap;
        // 源是 nor_gl（OpenGL 约定，绿向上），UE 的 SAMPLERTYPE_Normal 是 DirectX 约定
        // （绿向下）。不翻的话高光会与凹凸方向拧着。
        Texture->bFlipGreenChannel = true;
        break;
    case Channel::Roughness:
        Texture->SRGB = false;
        Texture->CompressionSettings = TC_Grayscale;
        Texture->LODGroup = TEXTUREGROUP_Cinematic;
        Texture->bFlipGreenChannel = false;
        break;
    }
    Texture->MaxTextureSize = 4096;
    // 常驻不流式：地形铺满整个视野，流式加载只会带来迟滞。
    Texture->NeverStream = true;
    Texture->PostEditChange();
    if (!SaveAsset(Texture))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save texture: %s"), *AssetName);
        return nullptr;
    }
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("MACRO_TEXTURE %s <- %s (%dx%d, sRGB=%s, compression=%d, flipGreen=%s)"),
        *AssetName, *Source, Texture->GetImportedSize().X, Texture->GetImportedSize().Y,
        Texture->SRGB ? TEXT("true") : TEXT("false"),
        static_cast<int32>(Texture->CompressionSettings),
        Texture->bFlipGreenChannel ? TEXT("true") : TEXT("false"));
    return Texture;
}

UMaterial* CreateMaterial(
    const FString& AssetPath,
    UTexture2D* Albedo,
    UTexture2D* Normal,
    UTexture2D* Roughness,
    const Options& Settings)
{
    const FString AssetName = FPaths::GetBaseFilename(AssetPath);
    const FString PackageName = FPaths::GetPath(AssetPath);
    UMaterial* Material = LoadObject<UMaterial>(nullptr, *(AssetPath + TEXT(".") + AssetName));
    if (!Material)
    {
        UPackage* Package = CreatePackage(*AssetPath);
        UMaterialFactoryNew* Factory = NewObject<UMaterialFactoryNew>();
        Material = Cast<UMaterial>(Factory->FactoryCreateNew(
            UMaterial::StaticClass(), Package, FName(*AssetName),
            RF_Public | RF_Standalone | RF_Transactional, nullptr, GWarn));
        if (Material)
        {
            FAssetRegistryModule::AssetCreated(Material);
        }
    }
    if (!Material)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to create material %s"), *AssetPath);
        return nullptr;
    }

    Material->Modify();
    UMaterialEditingLibrary::DeleteAllMaterialExpressions(Material);

    // 地形没有可用的 UV0：ALandscape 的 UV0 是地形空间的 quad 坐标，与 actor scale 无关，
    // 而任务运行时 task_scene.py 会把 scale 从地图里烘焙的 [50,50,25] 覆盖成 [1,1,1]，
    // 两者的世界尺寸差 50 倍。按地形空间平铺的话，demo 态下 4K 贴图会被铺到 1397 m 上
    // 变成约 0.1 px/cm，彻底糊掉。改按世界位置投影，重复次数随尺寸变、但永远不会糊。
    UMaterialExpressionWorldPosition* WorldPosition =
        CreateExpression<UMaterialExpressionWorldPosition>(Material, -1400, 0);
    WorldPosition->WorldPositionShaderOffset = WPT_Default;

    UMaterialExpressionComponentMask* WorldXY =
        CreateExpression<UMaterialExpressionComponentMask>(Material, -1200, 0);
    WorldXY->R = true;
    WorldXY->G = true;
    WorldXY->B = false;
    WorldXY->A = false;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldPosition, TEXT(""), WorldXY, TEXT("Input"));

    UMaterialExpressionScalarParameter* Tile =
        CreateExpression<UMaterialExpressionScalarParameter>(Material, -1200, 200);
    Tile->ParameterName = TEXT("UV_TileCm");
    Tile->DefaultValue = static_cast<float>(Settings.TileCm);

    UMaterialExpressionDivide* WorldUV =
        CreateExpression<UMaterialExpressionDivide>(Material, -1000, 0);
    UMaterialEditingLibrary::ConnectMaterialExpressions(WorldXY, TEXT(""), WorldUV, TEXT("A"));
    UMaterialEditingLibrary::ConnectMaterialExpressions(Tile, TEXT(""), WorldUV, TEXT("B"));

    UMaterialExpressionTextureSample* AlbedoSample =
        CreateExpression<UMaterialExpressionTextureSample>(Material, -720, -240);
    AlbedoSample->Texture = Albedo;
    AlbedoSample->SamplerType = SAMPLERTYPE_Color;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldUV, TEXT(""), AlbedoSample, TEXT("Coordinates"));

    UMaterialExpressionTextureSample* NormalSample =
        CreateExpression<UMaterialExpressionTextureSample>(Material, -720, 80);
    NormalSample->Texture = Normal;
    NormalSample->SamplerType = SAMPLERTYPE_Normal;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldUV, TEXT(""), NormalSample, TEXT("Coordinates"));

    UMaterialExpressionTextureSample* RoughnessSample =
        CreateExpression<UMaterialExpressionTextureSample>(Material, -720, 400);
    RoughnessSample->Texture = Roughness;
    RoughnessSample->SamplerType = SAMPLERTYPE_LinearGrayscale;
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        WorldUV, TEXT(""), RoughnessSample, TEXT("Coordinates"));

    // 法线强度旋钮。默认 1.0 时 Multiply+Normalize 是恒等变换，零影响；
    // 调小即压平法线。旧地形材质在 HLSL 里明确做了 norm*float3(1,1,.25)，
    // 说明这套扫描法线偏强，所以留一个可调入口。
    UMaterialExpressionVectorParameter* NormalStrength =
        CreateExpression<UMaterialExpressionVectorParameter>(Material, -720, 640);
    NormalStrength->ParameterName = TEXT("NormalScale");
    NormalStrength->DefaultValue = FLinearColor(
        static_cast<float>(Settings.NormalScale),
        static_cast<float>(Settings.NormalScale), 1.0f, 1.0f);

    UMaterialExpressionMultiply* ScaledNormal =
        CreateExpression<UMaterialExpressionMultiply>(Material, -420, 160);
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        NormalSample, TEXT("RGB"), ScaledNormal, TEXT("A"));
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        NormalStrength, TEXT(""), ScaledNormal, TEXT("B"));

    UMaterialExpressionNormalize* NormalizedNormal =
        CreateExpression<UMaterialExpressionNormalize>(Material, -220, 160);
    UMaterialEditingLibrary::ConnectMaterialExpressions(
        ScaledNormal, TEXT(""), NormalizedNormal, TEXT("VectorInput"));

    // 与旧材质的分工正好相反，值得说明：旧材质的 MP_Roughness 恒为 1、
    // MP_Specular 取自法线贴图的 B 通道，那是因为源数据把 specular 打包进了法线。
    // 本贴图集没有 specular/ARM 打包图，所以粗糙度交还给粗糙度贴图、
    // specular 退回一个常数。月壤极度非镜面，取 0.35（旧的非 V2 材质用 0.25，UE 默认 0.5）。
    UMaterialExpressionScalarParameter* Specular =
        CreateExpression<UMaterialExpressionScalarParameter>(Material, -420, 460);
    Specular->ParameterName = TEXT("Specular_Level");
    Specular->DefaultValue = static_cast<float>(Settings.Specular);

    if (!UMaterialEditingLibrary::ConnectMaterialProperty(
            AlbedoSample, TEXT("RGB"), MP_BaseColor) ||
        !UMaterialEditingLibrary::ConnectMaterialProperty(
            NormalizedNormal, TEXT(""), MP_Normal) ||
        !UMaterialEditingLibrary::ConnectMaterialProperty(
            RoughnessSample, TEXT("R"), MP_Roughness) ||
        !UMaterialEditingLibrary::ConnectMaterialProperty(
            Specular, TEXT(""), MP_Specular))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to connect macro material graph"));
        return nullptr;
    }

    UMaterialEditingLibrary::RecompileMaterial(Material);
    Material->PostEditChange();
    if (!SaveAsset(Material))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save material %s"), *AssetPath);
        return nullptr;
    }
    UE_LOG(LogMoonTerrainImport, Display, TEXT("MACRO_MATERIAL %s tileCm=%.1f specular=%.3f"),
        *AssetPath, Settings.TileCm, Settings.Specular);
    return Material;
}

// MIC 承载「每张地图自己的参数」。母材质不动，以后加地图 = 新 MIC + 新地图资产。
UMaterialInstanceConstant* CreateInstance(
    UMaterial* Parent,
    const FString& AssetPath,
    const Options& Settings)
{
    const FString AssetName = FPaths::GetBaseFilename(AssetPath);
    UMaterialInstanceConstant* Instance =
        LoadObject<UMaterialInstanceConstant>(nullptr, *(AssetPath + TEXT(".") + AssetName));
    if (!Instance)
    {
        UPackage* Package = CreatePackage(*AssetPath);
        UMaterialInstanceConstantFactoryNew* Factory =
            NewObject<UMaterialInstanceConstantFactoryNew>();
        Factory->InitialParent = Parent;
        Instance = Cast<UMaterialInstanceConstant>(Factory->FactoryCreateNew(
            UMaterialInstanceConstant::StaticClass(), Package, FName(*AssetName),
            RF_Public | RF_Standalone | RF_Transactional, nullptr, GWarn));
        if (Instance)
        {
            FAssetRegistryModule::AssetCreated(Instance);
        }
    }
    if (!Instance)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to create instance %s"), *AssetPath);
        return nullptr;
    }

    Instance->Modify();
    Instance->SetParentEditorOnly(Parent);
    UMaterialEditingLibrary::SetMaterialInstanceScalarParameterValue(
        Instance, TEXT("UV_TileCm"), static_cast<float>(Settings.TileCm));
    UMaterialEditingLibrary::SetMaterialInstanceScalarParameterValue(
        Instance, TEXT("Specular_Level"), static_cast<float>(Settings.Specular));
    UMaterialEditingLibrary::SetMaterialInstanceVectorParameterValue(
        Instance, TEXT("NormalScale"), FLinearColor(
            static_cast<float>(Settings.NormalScale),
            static_cast<float>(Settings.NormalScale), 1.0f, 1.0f));
    Instance->PostEditChange();
    if (!SaveAsset(Instance))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save instance %s"), *AssetPath);
        return nullptr;
    }
    return Instance;
}

ALandscape* FindLandscape(UWorld* World, int32& OutCount)
{
    ALandscape* Landscape = nullptr;
    OutCount = 0;
    for (TActorIterator<ALandscape> It(World); It; ++It)
    {
        Landscape = *It;
        ++OutCount;
    }
    return Landscape;
}

// 换皮 + 另存。不动高度数据、不动 transform、不动源地图与源材质。
bool Run(const FString& Params)
{
    Options Settings;
    FString Value;
    if (FParse::Value(*Params, TEXT("SrcMap="), Value) && !Value.IsEmpty()) Settings.SrcMap = Value;
    if (FParse::Value(*Params, TEXT("DstMap="), Value) && !Value.IsEmpty()) Settings.DstMap = Value;
    if (FParse::Value(*Params, TEXT("TextureDest="), Value) && !Value.IsEmpty()) Settings.TextureDest = Value;
    if (FParse::Value(*Params, TEXT("MaterialAsset="), Value) && !Value.IsEmpty()) Settings.Material = Value;
    if (FParse::Value(*Params, TEXT("InstanceAsset="), Value) && !Value.IsEmpty()) Settings.Instance = Value;
    if (FParse::Value(*Params, TEXT("TextureRoot="), Value) && !Value.IsEmpty()) Settings.TextureRoot = Value;
    FParse::Value(*Params, TEXT("TileCm="), Settings.TileCm);
    FParse::Value(*Params, TEXT("Specular="), Settings.Specular);
    FParse::Value(*Params, TEXT("NormalScale="), Settings.NormalScale);

    if (Settings.DstMap == Settings.SrcMap)
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("Refusing to overwrite the source map; -DstMap must differ from -SrcMap (%s)"),
            *Settings.SrcMap);
        return 12;
    }

    const FString SourceDirectory = MoonPaths::File(*Settings.TextureRoot);
    UTexture2D* Albedo = ImportTexture(SourceDirectory,
        TEXT("moon_macro_01_diff_4k.jpg"), TEXT("T_MoonMacro01_A"), Settings.TextureDest, Channel::Albedo);
    UTexture2D* Normal = ImportTexture(SourceDirectory,
        TEXT("moon_macro_01_nor_gl_4k.exr"), TEXT("T_MoonMacro01_N"), Settings.TextureDest, Channel::Normal);
    UTexture2D* Roughness = ImportTexture(SourceDirectory,
        TEXT("moon_macro_01_rough_4k.exr"), TEXT("T_MoonMacro01_R"), Settings.TextureDest, Channel::Roughness);
    if (!Albedo || !Normal || !Roughness)
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("Aborting before touching any map: texture import incomplete"));
        return 4;
    }

    UMaterial* Material = CreateMaterial(Settings.Material, Albedo, Normal, Roughness, Settings);
    if (!Material)
    {
        return 5;
    }
    UMaterialInstanceConstant* Instance = CreateInstance(Material, Settings.Instance, Settings);
    if (!Instance)
    {
        return 5;
    }

    UWorld* World = UEditorLoadingAndSavingUtils::LoadMap(Settings.SrcMap);
    if (!World)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to load source map %s"), *Settings.SrcMap);
        return 6;
    }
    int32 LandscapeCount = 0;
    ALandscape* Landscape = FindLandscape(World, LandscapeCount);
    if (!Landscape || LandscapeCount != 1)
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("Expected exactly one ALandscape in %s, found %d"),
            *Settings.SrcMap, LandscapeCount);
        return 7;
    }

    // 只改材质。高度数据、transform、组件布局、岩石与 Go2 部件全部原样带过去。
    Landscape->Modify();
    Landscape->LandscapeMaterial = Instance;
    // 不做这一步的话，每个 ULandscapeComponent 缓存的 MaterialInstanceConstant
    // 不会刷新，视口里仍是旧材质。MoonSceneV2.inl:130 也是这么做的。
    Landscape->UpdateAllComponentMaterialInstances(true);

    if (!UEditorLoadingAndSavingUtils::SaveMap(World, Settings.DstMap))
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("Failed to save map %s"), *Settings.DstMap);
        return 8;
    }
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("MACRO_MAP_SUCCESS src=%s dst=%s material=%s tileCm=%.1f"),
        *Settings.SrcMap, *Settings.DstMap, *Settings.Instance, Settings.TileCm);
    return 0;
}

bool Validate(const FString& Params)
{
    Options Settings;
    FString Value;
    if (FParse::Value(*Params, TEXT("DstMap="), Value) && !Value.IsEmpty()) Settings.DstMap = Value;
    if (FParse::Value(*Params, TEXT("TextureDest="), Value) && !Value.IsEmpty()) Settings.TextureDest = Value;
    if (FParse::Value(*Params, TEXT("InstanceAsset="), Value) && !Value.IsEmpty()) Settings.Instance = Value;

    const FString MapFilename = FPackageName::LongPackageNameToFilename(
        Settings.DstMap, FPackageName::GetMapPackageExtension());
    UWorld* World = UEditorLoadingAndSavingUtils::LoadMap(MapFilename);
    if (!World)
    {
        UE_LOG(LogMoonTerrainImport, Error, TEXT("VALIDATION: failed to load map %s"), *MapFilename);
        return false;
    }

    int32 LandscapeCount = 0;
    ALandscape* Landscape = FindLandscape(World, LandscapeCount);
    if (!Landscape || LandscapeCount != 1)
    {
        UE_LOG(LogMoonTerrainImport, Error,
            TEXT("VALIDATION: expected exactly one ALandscape, found %d"), LandscapeCount);
        return false;
    }

    TArray<ULandscapeComponent*> Components;
    Landscape->GetComponents(Components);
    int32 MinSectionX = MAX_int32;
    int32 MinSectionY = MAX_int32;
    int32 MaxVertexX = MIN_int32;
    int32 MaxVertexY = MIN_int32;
    bool bLayoutValid = Components.Num() == 22 * 22;
    for (const ULandscapeComponent* Component : Components)
    {
        if (!Component)
        {
            bLayoutValid = false;
            continue;
        }
        MinSectionX = FMath::Min(MinSectionX, Component->SectionBaseX);
        MinSectionY = FMath::Min(MinSectionY, Component->SectionBaseY);
        MaxVertexX = FMath::Max(MaxVertexX, Component->SectionBaseX + Component->ComponentSizeQuads);
        MaxVertexY = FMath::Max(MaxVertexY, Component->SectionBaseY + Component->ComponentSizeQuads);
        bLayoutValid &= Component->ComponentSizeQuads == QuadsPerSection;
        bLayoutValid &= Component->NumSubsections == SectionsPerComponent;
        bLayoutValid &= Component->SubsectionSizeQuads == QuadsPerSection;
    }
    const int32 ResolutionX = MaxVertexX - MinSectionX + 1;
    const int32 ResolutionY = MaxVertexY - MinSectionY + 1;
    bLayoutValid &= MinSectionX == 0 && MinSectionY == 0;
    bLayoutValid &= ResolutionX == Resolution && ResolutionY == Resolution;

    // transform 必须仍是 scene.json 里烘焙的那一份——这是「没碰几何」的直接证据。
    const TSharedPtr<FJsonObject> Scene = MoonSceneV2::ReadObject(TEXT("scene.json"));
    const FVector ExpectedLocation = MoonSceneV2::Vec(Scene->GetArrayField(TEXT("location")));
    const FVector ExpectedScale = MoonSceneV2::Vec(Scene->GetArrayField(TEXT("scale")));
    const FVector Location = Landscape->GetActorLocation();
    const FVector Scale = Landscape->GetActorScale3D();
    const bool bTransformValid = Location.Equals(ExpectedLocation, 0.01)
        && Scale.Equals(ExpectedScale, 0.000001);

    UMaterialInterface* Assigned = Landscape->GetLandscapeMaterial();
    const FString ExpectedMaterial =
        Settings.Instance + TEXT(".") + FPaths::GetBaseFilename(Settings.Instance);
    const bool bMaterialValid = Assigned && Assigned->GetPathName() == ExpectedMaterial;

    UTexture2D* Albedo = LoadObject<UTexture2D>(nullptr,
        *(Settings.TextureDest + TEXT("/T_MoonMacro01_A.T_MoonMacro01_A")));
    UTexture2D* Normal = LoadObject<UTexture2D>(nullptr,
        *(Settings.TextureDest + TEXT("/T_MoonMacro01_N.T_MoonMacro01_N")));
    UTexture2D* Roughness = LoadObject<UTexture2D>(nullptr,
        *(Settings.TextureDest + TEXT("/T_MoonMacro01_R.T_MoonMacro01_R")));
    const bool bAlbedoValid = Albedo && Albedo->GetImportedSize() == FIntPoint(4096, 4096)
        && Albedo->SRGB && Albedo->CompressionSettings == TC_BC7;
    // 绿通道翻转是这里唯一无法从「属性都设对了」推断出来的项：
    // 忘了翻的话渲染会静默地错（高光与凹凸方向拧着），所以单独断言。
    const bool bNormalValid = Normal && Normal->GetImportedSize() == FIntPoint(4096, 4096)
        && !Normal->SRGB && Normal->CompressionSettings == TC_Normalmap
        && Normal->bFlipGreenChannel;
    const bool bRoughnessValid = Roughness && Roughness->GetImportedSize() == FIntPoint(4096, 4096)
        && !Roughness->SRGB && Roughness->CompressionSettings == TC_Grayscale;

    bool bReferencesAlbedo = false;
    bool bReferencesNormal = false;
    bool bReferencesRoughness = false;
    if (Assigned)
    {
        for (UObject* Referenced : Assigned->GetReferencedTextures())
        {
            bReferencesAlbedo |= Referenced == Albedo;
            bReferencesNormal |= Referenced == Normal;
            bReferencesRoughness |= Referenced == Roughness;
        }
    }
    const bool bReferencesValid = bReferencesAlbedo && bReferencesNormal && bReferencesRoughness;

    // 逐点比对高度场。这条同时证明两件事：没做位移、副本没损坏几何。
    TArray<uint16> Expected;
    bool bHeightsValid = LoadHeightData(
        MoonPaths::File(TEXT("ue/import_data/Landscape_1_2795x2795.r16")), Expected);
    int64 Mismatches = -1;
    if (bHeightsValid)
    {
        ULandscapeInfo* Info = Landscape->GetLandscapeInfo();
        if (!Info)
        {
            bHeightsValid = false;
        }
        else
        {
            TArray<uint16> Loaded;
            Loaded.SetNumUninitialized(Resolution * Resolution);
            FLandscapeEditDataInterface Edit(Info, false);
            Edit.GetHeightDataFast(0, 0, Resolution - 1, Resolution - 1, Loaded.GetData(), Resolution);
            Mismatches = 0;
            for (int32 Index = 0; Index < Loaded.Num(); ++Index)
            {
                if (Loaded[Index] != Expected[Index])
                {
                    ++Mismatches;
                }
            }
            bHeightsValid = Mismatches == 0;
        }
    }

    int32 Go2VisualCount = 0;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        if (It->ActorHasTag(TEXT("Go2Visual")))
        {
            ++Go2VisualCount;
        }
    }
    const bool bGo2Valid = Go2VisualCount == 33;

    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_GEOMETRY: actors=%d components=%d resolution=%dx%d layout=%s"),
        LandscapeCount, Components.Num(), ResolutionX, ResolutionY,
        bLayoutValid ? TEXT("ok") : TEXT("bad"));
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_TRANSFORM: location_cm=(%.6f,%.6f,%.6f) expected=(%.6f,%.6f,%.6f) scale=(%.6f,%.6f,%.6f) ok=%s"),
        Location.X, Location.Y, Location.Z,
        ExpectedLocation.X, ExpectedLocation.Y, ExpectedLocation.Z,
        Scale.X, Scale.Y, Scale.Z, bTransformValid ? TEXT("true") : TEXT("false"));
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_HEIGHTS: samples=%d mismatches=%lld ok=%s"),
        Expected.Num(), Mismatches, bHeightsValid ? TEXT("true") : TEXT("false"));
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_MATERIAL: assigned=%s expected=%s refs=(albedo:%s normal:%s rough:%s)"),
        Assigned ? *Assigned->GetPathName() : TEXT("None"), *ExpectedMaterial,
        bReferencesAlbedo ? TEXT("yes") : TEXT("no"),
        bReferencesNormal ? TEXT("yes") : TEXT("no"),
        bReferencesRoughness ? TEXT("yes") : TEXT("no"));
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_TEXTURES: albedo=%s normal=%s rough=%s"),
        bAlbedoValid ? TEXT("ok") : TEXT("bad"),
        bNormalValid ? TEXT("ok") : TEXT("bad"),
        bRoughnessValid ? TEXT("ok") : TEXT("bad"));
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_GO2: parts=%d ok=%s"), Go2VisualCount,
        bGo2Valid ? TEXT("true") : TEXT("false"));

    const bool bPass = bLayoutValid && bTransformValid && bHeightsValid
        && bMaterialValid && bReferencesValid
        && bAlbedoValid && bNormalValid && bRoughnessValid && bGo2Valid;
    UE_LOG(LogMoonTerrainImport, Display,
        TEXT("VALIDATION_MATERIAL_MATCH: %s"), bMaterialValid ? TEXT("true") : TEXT("false"));
    UE_LOG(LogMoonTerrainImport, Display, TEXT("MACRO_MAP_VALIDATION: %s"),
        bPass ? TEXT("PASS") : TEXT("FAIL"));
    return bPass;
}
} // namespace MoonMacro
