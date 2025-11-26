
// TransferGameBase.cpp

#include "TransferGameBase.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

ATransferGameBase::ATransferGameBase()
{
    // 可以在构造时设置默认值
}

void ATransferGameBase::BeginPlay()
{
    Super::BeginPlay();

    LoadConfigJson();

    UE_LOG(LogTemp, Warning, TEXT("Config Loaded: IP=%s Port=%d"),*GameConfig.ServerIP, GameConfig.Port);
}

void ATransferGameBase::LoadConfigJson()
{
    FString FilePath = FPaths::ProjectDir() / TEXT("Config/GameConfig.json");

    FString JsonRaw;
    if (!FFileHelper::LoadFileToString(JsonRaw, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to load config file: %s"), *FilePath);
        return;
    }

    TSharedPtr<FJsonObject> JsonObj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonRaw);

    if (!FJsonSerializer::Deserialize(Reader, JsonObj) || !JsonObj.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Json Deserialize failed!"));
        return;
    }

    // 解析字段
    GameConfig.ServerIP = JsonObj->GetStringField(TEXT("ServerIP"));
    GameConfig.Port = JsonObj->GetIntegerField(TEXT("Port"));
    GameConfig.UseGPU = JsonObj->GetBoolField(TEXT("UseGPU"));
    GameConfig.WindowWidth = JsonObj->GetIntegerField(TEXT("WindowWidth"));
    GameConfig.WindowHeight = JsonObj->GetIntegerField(TEXT("WindowHeight"));
}
