#pragma once

#include "Commandlets/Commandlet.h"
#include "MoonTerrainImportCommandlet.generated.h"

UCLASS()
class MOONTERRAINIMPORTER_API UMoonTerrainImportCommandlet : public UCommandlet
{
    GENERATED_BODY()

public:
    UMoonTerrainImportCommandlet();
    virtual int32 Main(const FString& Params) override;
};
