using UnrealBuildTool;

public class MoonTerrainImporter : ModuleRules
{
    public MoonTerrainImporter(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "UnrealEd",
            "Landscape",
            "Foliage",
            "Json",
            "MeshDescription",
            "StaticMeshDescription",
            "MaterialEditor"
            ,"Slate", "SlateCore", "InputCore", "Sockets", "Networking", "PythonScriptPlugin"
        });
    }
}
