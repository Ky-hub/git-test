// TransferGameBase.h

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "TransferGameBase.generated.h"

USTRUCT(BlueprintType)
struct FGameConfigData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString ServerIP;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Port = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool UseGPU = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 WindowWidth = 1280;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 WindowHeight = 720;
};

UCLASS()
class METAHUMANBYD_API ATransferGameBase : public AGameModeBase
{
    GENERATED_BODY()

public:
    ATransferGameBase();

protected:
    virtual void BeginPlay() override;

private:
    void LoadConfigJson();

public:
    UPROPERTY(BlueprintReadOnly)
    FGameConfigData GameConfig;
};
