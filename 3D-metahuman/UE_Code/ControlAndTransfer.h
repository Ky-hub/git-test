// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneCaptureComponent2D.h" 
#include "Engine/TextureRenderTarget2D.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"

#include "HAL/RunnableThread.h"
#include "HAL/CriticalSection.h"

#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

#include "ControlAndTransfer.generated.h"

class AControlAndTransfer;

class FDataReceiverRunnable : public FRunnable
{
public:
	FSocket* ListenSocket = nullptr;
	FSocket* ClientSocket = nullptr;
	TArray<TArray<double>>* BodyMotionData = nullptr;
	TArray<TArray<double>>* FaceMotionData = nullptr;
	FThreadSafeBool bRunThread = true;
	int32* BodyFrameIndex = nullptr;
	int32* FaceFrameIndex = nullptr;
	bool* b_getData;
	int32* fps;
	int32* FrameLength;
	int32 LengthLimit = 1500;
	AControlAndTransfer* OwnerActor = nullptr;

	FCriticalSection DataLock;

public:
	FDataReceiverRunnable(
		AControlAndTransfer* Owner,
		FSocket* InSocket,
		TArray<TArray<double>>* InBodyData,
		TArray<TArray<double>>* InFaceData,
		int32* InBodyIndex,
		int32* InFaceIndex,
		bool* b_getData_in,
		int32* in_fps,
		int32* InFrameLength
		);

	virtual uint32 Run() override;
	void ParseMotionData(const FString& JsonStr);
	virtual void Stop() override;
};


UCLASS()
class METAHUMANBYD_API AControlAndTransfer : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AControlAndTransfer();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:	
	// ----------------------------- ----------------------------
	virtual void Tick(float DeltaTime) override;

	void CaptureAndEncodeFrame();
	bool InitTCPServer();
	bool InitDataProcessServer();
	void SendData(const TArray<uint8>& Data);
	void SentDataWithTCP(const TArray<uint8>& Data,int32 width, int32 height);
	void UpdateCaptureFullSettings(
		int32 NewWidth,
		int32 NewHeight,
		float NewFOV,
		FVector NewLocation,
		FRotator NewRotation
	);

	// ----------------------------- -----------------------------
	void SetMotionData();





public:
	// 编辑器可设置、蓝图可读写
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture")
	USceneCaptureComponent2D* SceneCapture;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Video Capture")
	UTextureRenderTarget2D* RenderTarget;


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TCP Settings")
	FString DateServerIP = TEXT("127.0.0.1");

	/** ✅ 编辑器中可修改的远程端口 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TCP Settings")
	int32 DataSeerverPort = 7081;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TCP Settings")
	FString UnrealServerIP = TEXT("127.0.0.1");

	/** ✅ 编辑器中可修改的远程端口 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TCP Settings")
	int32 UnrealServerPort = 8083;

	FSocket* DataClientSocket = nullptr;
	bool b_connectDataServer = false;

	FSocket* ListenSocket = nullptr;
	int32 FrameCounter = 0;
	TArray<FColor> Bitmap;
	TArray<uint8> ByteData;


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Target")
	AActor* TargetActor;  // 你想绑定的场景 Actor

	USkeletalMeshComponent* BodyMesh;
	USkeletalMeshComponent* FaceMesh;
	UAnimInstance* BodyAnimBPInstance;
	UAnimInstance* FaceAnimBPInstance;

	TArray<TArray<double>> BodyMotionData;
	TArray<TArray<double>> FaceMotionData;
    int32 bodyFrameIndex = -1;
    int32 faceFrameIndex = -1;
	int32 currentFrameIndex = -1;
	int32 frameLength = 0;
	int32 cacheFrameLength = 1500;
	int32 fps = 30;
	bool b_getData = false;

    FRunnableThread* ReceiverThread = nullptr;
    FDataReceiverRunnable* ReceiverRunnable = nullptr;

};
