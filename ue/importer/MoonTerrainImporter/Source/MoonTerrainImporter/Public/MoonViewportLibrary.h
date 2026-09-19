#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "MoonViewportLibrary.generated.h"

class UStaticMesh;

// Editor-specific glue for the existing demo, not a general simulation API.
UCLASS()
class MOONTERRAINIMPORTER_API UMoonViewportLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="MoonDemo")
    static void ApplyCamera(const TArray<double>& Frame);
    UFUNCTION(BlueprintCallable, Category="MoonDemo")
    static TArray<double> ReadCamera();
    UFUNCTION(BlueprintCallable, Category="MoonDemo")
    static TArray<FVector> ProjectPoints(const TArray<FVector>& Points);
    UFUNCTION(BlueprintCallable, Category="MoonDemo")
    static UStaticMesh* CreateTaskMesh(const FString& JsonFile);
    UFUNCTION(BlueprintCallable, Category="MoonSensors")
    static bool CaptureTaskRGBD(const FString& File, FVector Position, FRotator Rotation,
                                int32 Width, int32 Height, float HorizontalFov);
    static void ReleaseCamera();
};
