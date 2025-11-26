// Fill out your copyright notice in the Description page of Project Settings.


#include "ControlAndTransfer.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "BodyConInstance.h"
#include "FaceConInstance.h"

// Sets default values
AControlAndTransfer::AControlAndTransfer()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
    if (!SceneCapture)
    {
        SceneCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
        RootComponent = SceneCapture;
    }

    if (!RenderTarget)
    {
        RenderTarget = NewObject<UTextureRenderTarget2D>();
        RenderTarget->InitAutoFormat(1024, 1024);
        RenderTarget->ClearColor = FLinearColor::Black;
        RenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;  // 指定格式
        RenderTarget->UpdateResourceImmediate();

        if (SceneCapture)
            SceneCapture->TextureTarget = RenderTarget;

    }




}

// Called when the game starts or when spawned
void AControlAndTransfer::BeginPlay()
{
    Super::BeginPlay();

    if (!TargetActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("❌ 未指定 TargetActor，无法初始化绑定。"));
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("🎯 尝试绑定目标 Actor: %s"), *TargetActor->GetName());

    // 1️⃣ 遍历目标 Actor 的所有组件
    TArray<USkeletalMeshComponent*> SkeletalComps;
    TargetActor->GetComponents<USkeletalMeshComponent>(SkeletalComps);

    if (SkeletalComps.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("⚠️ 目标 Actor 没有 SkeletalMeshComponent。"));
        return;
    }

    // 2️⃣ 自动识别 Body / Face Mesh
    for (USkeletalMeshComponent* Comp : SkeletalComps)
    {
        FString Name = Comp->GetName();

        if (Name.Contains(TEXT("Body"), ESearchCase::IgnoreCase))
        {
            BodyMesh = Comp;
            UE_LOG(LogTemp, Log, TEXT("✅ 找到 BodyMesh: %s"), *Name);
        }
        else if (Name.Contains(TEXT("Face"), ESearchCase::IgnoreCase))
        {
            FaceMesh = Comp;
            UE_LOG(LogTemp, Log, TEXT("✅ 找到 FaceMesh: %s"), *Name);
        }
    }

    // 3️⃣ 如果找到了 Mesh，就记录它们当前使用的 AnimBP
    if (BodyMesh)
    {
        if (UAnimInstance* Anim = BodyMesh->GetAnimInstance())
        {
            BodyAnimBPInstance = Anim;
            UE_LOG(LogTemp, Log, TEXT("📦 Body 动画实例类: %s"), *Anim->GetClass()->GetName());
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("⚠️ BodyMesh 没有关联动画实例！"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("⚠️ BodyMesh 未设置！"));
    }

    if (FaceMesh)
    {
        if (UAnimInstance* Anim = FaceMesh->GetAnimInstance())
        {
            FaceAnimBPInstance = Anim;
            UE_LOG(LogTemp, Log, TEXT("📦 Face 动画实例类: %s"), *Anim->GetClass()->GetName());
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("⚠️ FaceMesh 没有关联动画实例！"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("⚠️ FaceMesh 未设置！"));
    }


    if (RenderTarget)
    {
        int32 Width = RenderTarget->SizeX;
        int32 Height = RenderTarget->SizeY;

        // 一次性分配数组
        Bitmap.SetNumUninitialized(Width * Height);
        ByteData.SetNumUninitialized(Width * Height * 4);
    }

    BodyMotionData.SetNum(cacheFrameLength);
    for (int32 i = 0; i < cacheFrameLength; ++i)
    {
        BodyMotionData[i].SetNumZeroed(165);
    }

    // 初始化 FaceMotionData 为 1500x136
    FaceMotionData.SetNum(cacheFrameLength);
    for (int32 i = 0; i < cacheFrameLength; ++i)
    {
        FaceMotionData[i].SetNumZeroed(136);
    }

    InitTCPServer();


}

// Called every frame
void AControlAndTransfer::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
    if (b_getData)
    {
        UE_LOG(LogTemp, Warning, TEXT("b_getData == true, calling SetMotionData()"));

        FScopeLock Lock(&ReceiverRunnable->DataLock);
        SetMotionData();
        b_getData = false;
    }
	CaptureAndEncodeFrame();
}

void AControlAndTransfer::CaptureAndEncodeFrame()
{

    if (!RenderTarget || !SceneCapture)
    {
        UE_LOG(LogTemp, Warning, TEXT("⚠️ RenderTarget 或 SceneCapture 未设置"));
        return;
    }

    // 获取渲染目标资源
    FTextureRenderTargetResource* RTResource = RenderTarget->GameThread_GetRenderTargetResource();
    if (!RTResource)
    {
        UE_LOG(LogTemp, Warning, TEXT("RenderTarget resource not available."));
        return;
    }

    // 读取像素（BGRA 格式）
    bool bReadSuccess = RTResource->ReadPixels(Bitmap);
    if (!bReadSuccess || Bitmap.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to read pixels from RenderTarget."));
        return;
    }

    int32 Width = RenderTarget->SizeX;
    int32 Height = RenderTarget->SizeY;

    // 将 FColor 数组转为字节流
    for (int32 i = 0; i < Bitmap.Num(); i++)
    {
        const FColor& Color = Bitmap[i];
        int32 Offset = i * 4;
        ByteData[Offset + 0] = Color.R;
        ByteData[Offset + 1] = Color.G;
        ByteData[Offset + 2] = Color.B;
        ByteData[Offset + 3] = 255 - Color.A; // Alpha 固定 255
    }

    // 调用发送函数（HTTP上传）
    //SendData(ByteData);

    InitDataProcessServer();
    SentDataWithTCP(ByteData,Width,Height);

    FrameCounter++;

}

bool AControlAndTransfer::InitTCPServer()
{
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem) return false;

    // 创建 TCP Socket（监听）
    ListenSocket = SocketSubsystem->CreateSocket(NAME_Stream, TEXT("VideoTCP_Server"), false);
    if (!ListenSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create TCP server socket"));
        return false;
    }

    // 转换 IP 地址
    TSharedRef<FInternetAddr> Addr = SocketSubsystem->CreateInternetAddr();
    bool bIsValid;
    Addr->SetIp(*UnrealServerIP, bIsValid);  // UE 监听的 IP
    Addr->SetPort(UnrealServerPort);         // UE 监听的端口

    if (!bIsValid)
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid IP Address: %s"), *UnrealServerIP);
        return false;
    }

    // 绑定端口
    if (!ListenSocket->Bind(*Addr))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to bind TCP server socket"));
        return false;
    }

    // 开始监听，最大等待连接数 1
    if (!ListenSocket->Listen(1))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to listen on TCP server socket"));
        return false;
    }

    ListenSocket->SetNonBlocking(true);
    ListenSocket->SetNoDelay(true);

    UE_LOG(LogTemp, Log, TEXT("TCP Server listening on %s:%d"), *UnrealServerIP, UnrealServerPort);

    // 创建并启动接收线程
    ReceiverRunnable = new FDataReceiverRunnable(
        this,
        ListenSocket,
        &BodyMotionData,
        &FaceMotionData,
        &bodyFrameIndex,
        &faceFrameIndex,
        &b_getData,
        &fps,
        &frameLength
    );

    ReceiverThread = FRunnableThread::Create(ReceiverRunnable, TEXT("DataReceiverThread"));

    return true;
}

void AControlAndTransfer::SendData(const TArray<uint8>& Data)
{
    FHttpModule* Http = &FHttpModule::Get();
    if (!Http) return;

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = Http->CreateRequest();
    Request->SetURL(FString::Printf(TEXT("http://%s:%d/upload"), *DateServerIP, DataSeerverPort));
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/octet-stream"));

    Request->SetHeader(TEXT("X-Frame-Counter"), FString::FromInt(FrameCounter));
    Request->SetContent(Data);

    Request->OnProcessRequestComplete().BindLambda([](FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bSuccess) {
        if (!bSuccess || !Resp.IsValid())
        {
            UE_LOG(LogTemp, Warning, TEXT("HTTP upload failed"));
            return;
        }
        UE_LOG(LogTemp, Log, TEXT("HTTP upload OK: %d"), Resp->GetResponseCode());
        });

    Request->ProcessRequest();
}

void AControlAndTransfer::SetMotionData()
{
    UE_LOG(LogTemp, Warning, TEXT("SetMotionData() called."));

    // 检查 FaceAnimBPInstance 是否存在
    if (FaceAnimBPInstance)
    {
        UE_LOG(LogTemp, Warning, TEXT("FaceAnimBPInstance is valid."));
        UFaceConInstance* FaceAnim = Cast<UFaceConInstance>(FaceAnimBPInstance);
        if (FaceAnim)
        {
            UE_LOG(LogTemp, Warning, TEXT("Cast to UFaceConInstance succeeded."));
            UE_LOG(LogTemp, Warning, TEXT("Setting FaceMotionData and BodyMotionData, fps = %d"), fps);
            FaceAnim->SetFaceMotionData(FaceMotionData, fps, frameLength);
            FaceAnim->SetBodyMotionData(BodyMotionData, fps, frameLength);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Cast to UFaceConInstance failed."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("FaceAnimBPInstance is NULL."));
    }

    // 检查 BodyAnimBPInstance 是否存在
    if (BodyAnimBPInstance)
    {
        UE_LOG(LogTemp, Warning, TEXT("BodyAnimBPInstance is valid."));
        UBodyConInstance* BodyAnim = Cast<UBodyConInstance>(BodyAnimBPInstance);
        if (BodyAnim)
        {
            UE_LOG(LogTemp, Warning, TEXT("Cast to UBodyConInstance succeeded."));
            UE_LOG(LogTemp, Warning, TEXT("Setting BodyMotionData, fps = %d"), fps);
            BodyAnim->SetBodyMotionData(BodyMotionData, fps, frameLength);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Cast to UBodyConInstance failed."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("BodyAnimBPInstance is NULL."));
    }

}

bool AControlAndTransfer::InitDataProcessServer()
{
    if (b_connectDataServer && DataClientSocket)
    {
        return true; // 已连接
    }

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem)
    {
        UE_LOG(LogTemp, Error, TEXT("Cannot get SocketSubsystem"));
        return false;
    }

    // 创建 TCP socket
    DataClientSocket = SocketSubsystem->CreateSocket(NAME_Stream, TEXT("TCP_DataClientSocket"), false);
    if (!DataClientSocket)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create TCP socket"));
        return false;
    }

    DataClientSocket->SetNonBlocking(true);
    DataClientSocket->SetNoDelay(true);

    // 创建服务器地址
    TSharedRef<FInternetAddr> ServerAddr = SocketSubsystem->CreateInternetAddr();
    bool bIsValid;
    ServerAddr->SetIp(*DateServerIP, bIsValid);
    ServerAddr->SetPort(DataSeerverPort);

    if (!bIsValid)
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid server IP: %s"), *DateServerIP);
        SocketSubsystem->DestroySocket(DataClientSocket);
        DataClientSocket = nullptr;
        return false;
    }

    // 尝试连接
    bool bConnected = DataClientSocket->Connect(*ServerAddr);
    if (!bConnected)
    {
        UE_LOG(LogTemp, Warning, TEXT("Cannot connect to server %s:%d"), *DateServerIP, DataSeerverPort);
        SocketSubsystem->DestroySocket(DataClientSocket);
        DataClientSocket = nullptr;
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("Connected to data server %s:%d"), *DateServerIP, DataSeerverPort);

    b_connectDataServer = true; // 标记已连接
    return true;
}

void AControlAndTransfer::SentDataWithTCP(const TArray<uint8>& PixelData, int32 Width, int32 Height)
{
    if (!DataClientSocket || !b_connectDataServer || PixelData.Num() == 0) return;

    uint32 FrameLength = PixelData.Num();
    auto ToNetOrder = [](uint32 Val) {
        return ((Val & 0xFF) << 24) | ((Val & 0xFF00) << 8) | ((Val & 0xFF0000) >> 8) | ((Val & 0xFF000000) >> 24);
        };

    uint32 NetFrameLen = ToNetOrder(FrameLength);
    uint32 NetWidth = ToNetOrder(Width);
    uint32 NetHeight = ToNetOrder(Height);

    TArray<uint8> SendBuffer;
    SendBuffer.Append(reinterpret_cast<uint8*>(&NetFrameLen), 4);
    SendBuffer.Append(reinterpret_cast<uint8*>(&NetWidth), 4);
    SendBuffer.Append(reinterpret_cast<uint8*>(&NetHeight), 4);
    SendBuffer.Append(PixelData);

    // 🔥 关键点 1：先等待可写
    if (!DataClientSocket->Wait(ESocketWaitConditions::WaitForWrite, FTimespan::FromMilliseconds(1)))
    {
        UE_LOG(LogTemp, Warning, TEXT("Socket not ready to write. Drop frame."));
        return;
    }

    // 🔥 关键点 2：Send 循环加重试上限
    int32 TotalSent = 0;
    int RetryCount = 0;
    const int MaxRetry = 20;

    while (TotalSent < SendBuffer.Num())
    {
        int32 BytesSent = 0;
        bool bSent = DataClientSocket->Send(
            SendBuffer.GetData() + TotalSent,
            SendBuffer.Num() - TotalSent,
            BytesSent
        );

        if (!bSent || BytesSent <= 0)
        {
            RetryCount++;
            if (RetryCount >= MaxRetry)
            {
                UE_LOG(LogTemp, Warning, TEXT("Send stalled. Drop frame."));
                return;
            }

            FPlatformProcess::Sleep(0.0005f);
            continue;
        }

        TotalSent += BytesSent;
    }
}

void AControlAndTransfer::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);

    // 停止线程
    if (ReceiverRunnable)
    {
        ReceiverRunnable->Stop();
        if (ReceiverThread)
        {
            ReceiverThread->WaitForCompletion();
            delete ReceiverThread;
            ReceiverThread = nullptr;
        }
        delete ReceiverRunnable;
        ReceiverRunnable = nullptr;
    }

    // 关闭 TCP socket
    if (ListenSocket)
    {
        ListenSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ListenSocket);
        ListenSocket = nullptr;
    }

    if (DataClientSocket)
    {
        DataClientSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ListenSocket);
        DataClientSocket = nullptr;
        b_connectDataServer = false;
    }

    UE_LOG(LogTemp, Log, TEXT("AControlAndTransfer::EndPlay called"));
}

void AControlAndTransfer::UpdateCaptureFullSettings(
    int32 NewWidth,
    int32 NewHeight,
    float NewFOV,
    FVector NewLocation,
    FRotator NewRotation
)
{
    if (!RenderTarget || !SceneCapture)
    {
        UE_LOG(LogTemp, Warning, TEXT("RenderTarget 或 SceneCapture 未设置！"));
        return;
    }

    // -------------------------------
    // 1. 修改 RenderTarget 分辨率
    // -------------------------------
    RenderTarget->ResizeTarget(NewWidth, NewHeight);
    RenderTarget->UpdateResourceImmediate(true);

    // -------------------------------
    // 2. 修改 SceneCapture 参数
    // -------------------------------
    SceneCapture->FOVAngle = NewFOV;
    SceneCapture->TextureTarget = RenderTarget;

    // 设置捕获源（你可以改成 HDR）
    SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    // 自动捕获开关
    SceneCapture->bCaptureEveryFrame = true;
    SceneCapture->bCaptureOnMovement = true;

    // -------------------------------
    // 3. 修改位置与旋转
    // -------------------------------
    SceneCapture->SetWorldLocation(NewLocation);
    SceneCapture->SetWorldRotation(NewRotation);

    // -------------------------------
    // 4. 强制 SceneCapture 立即刷新
    // -------------------------------
    SceneCapture->CaptureScene();

    UE_LOG(LogTemp, Warning,
        TEXT("SceneCapture 更新成功：分辨率 = %d x %d FOV=%.1f 位置=%s 旋转=%s"),
        NewWidth,
        NewHeight,
        NewFOV,
        *NewLocation.ToString(),
        *NewRotation.ToString()
    );
}


// =============================================================
// ----------- FDataReceiverRunnable Implementation ------------
// =============================================================
FDataReceiverRunnable::FDataReceiverRunnable(
    AControlAndTransfer* Owner,
    FSocket* InSocket,
    TArray<TArray<double>>* InBodyData,
    TArray<TArray<double>>* InFaceData,
    int32* InBodyIndex,
    int32* InFaceIndex,
    bool* b_getData_in,
    int32* in_fps,
    int32* InFrameLength)
    : OwnerActor(Owner),
    ListenSocket(InSocket),
    BodyMotionData(InBodyData),
    FaceMotionData(InFaceData),
    BodyFrameIndex(InBodyIndex),
    FaceFrameIndex(InFaceIndex),
    b_getData(b_getData_in),
    fps(in_fps),
    FrameLength(InFrameLength)
{
}

uint32 FDataReceiverRunnable::Run()
{
    TArray<uint8> ReceiveBuffer; // 接收缓冲区，用于拼接不完整消息

    while (bRunThread)
    {
        // 如果监听 socket 不存在，等待
        if (!ListenSocket)
        {
            FPlatformProcess::Sleep(0.1f);
            continue;
        }

        // --------- 接受客户端连接 ---------
        if (!ClientSocket)
        {
            TSharedRef<FInternetAddr> ClientAddr = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
            ClientSocket = ListenSocket->Accept(*ClientAddr, TEXT("UE_TCP_Client"));
            if (ClientSocket)
            {
                ClientSocket->SetNonBlocking(true);
                ClientSocket->SetNoDelay(true);
                UE_LOG(LogTemp, Log, TEXT("Client connected: %s"), *ClientAddr->ToString(true));
            }
        }

        // --------- 读取客户端数据 ---------
        if (ClientSocket)
        {
            uint32 PendingDataSize = 0;
            while (ClientSocket->HasPendingData(PendingDataSize) && PendingDataSize > 0)
            {
                TArray<uint8> TempBuffer;
                TempBuffer.SetNumUninitialized(PendingDataSize);

                int32 BytesRead = 0;
                if (ClientSocket->Recv(TempBuffer.GetData(), TempBuffer.Num(), BytesRead) && BytesRead > 0)
                {
                    // 拼接到缓冲区
                    ReceiveBuffer.Append(TempBuffer.GetData(), BytesRead);

                    // 循环处理完整消息（防止粘包）
                    while (ReceiveBuffer.Num() >= 4)
                    {
                        // ---- 读取4字节长度前缀（大端）----
                        uint32 MsgLen = 0;
                        FMemory::Memcpy(&MsgLen, ReceiveBuffer.GetData(), 4);
                        MsgLen = ((MsgLen & 0xFF) << 24) |
                            ((MsgLen & 0xFF00) << 8) |
                            ((MsgLen & 0xFF0000) >> 8) |
                            ((MsgLen & 0xFF000000) >> 24);

                        // ---- 检查是否收到完整消息 ----
                        if (static_cast<uint32>(ReceiveBuffer.Num()) >= 4 + MsgLen)
                        {
                            const uint8* JsonBytes = ReceiveBuffer.GetData() + 4;

                            // ---- 使用 FUTF8ToTCHAR 转换 UTF8 JSON ----
                            FString JsonString = FString(FUTF8ToTCHAR(reinterpret_cast<const ANSICHAR*>(JsonBytes), MsgLen));

                            UE_LOG(LogTemp, Verbose, TEXT("Received JSON length: %u, buffer: %d"), MsgLen, ReceiveBuffer.Num());

                            // ---- 解析数据 ----
                            
                            FScopeLock Lock(&DataLock);
                            ParseMotionData(JsonString);
                            

                            // ---- 清理缓冲 ----
                            ReceiveBuffer.RemoveAt(0, 4 + MsgLen, false);
                            *b_getData = true;
                        }
                        else
                        {
                            // 数据还没收全，等下一帧
                            break;
                        }
                    }
                }
            }

            // --------- 检查连接状态 ---------
            if (ClientSocket->GetConnectionState() != SCS_Connected)
            {
                UE_LOG(LogTemp, Warning, TEXT("Client disconnected"));
                ClientSocket->Close();
                ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ClientSocket);
                ClientSocket = nullptr;
                ReceiveBuffer.Empty();
            }
        }

        FPlatformProcess::Sleep(0.01f); // 控制循环频率
    }

    return 0;
}


void FDataReceiverRunnable::ParseMotionData(const FString& JsonStr)
{
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonStr);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
        return;


    // ----------------- BodyData -----------------
    const TArray<TSharedPtr<FJsonValue>>* BodyArray = nullptr;
    if (JsonObject->TryGetArrayField(TEXT("motion_pred"), BodyArray))
    {
        *BodyFrameIndex = -1;
        int32 FrameCount = BodyArray->Num();
        int32 ElementCount = FrameCount > 0 && (*BodyArray)[0]->Type == EJson::Array
            ? (*BodyArray)[0]->AsArray().Num() : 0;
        UE_LOG(LogTemp, Log, TEXT("BodyData shape: frames=%d, elements per frame=%d"), FrameCount, ElementCount);

        for (auto& FrameValue : *BodyArray)
        {
            if (FrameValue->Type == EJson::Array)
            {

                const TArray<TSharedPtr<FJsonValue>>& Frame = FrameValue->AsArray();
                int32 WriteIndex = (*BodyFrameIndex + 1) % LengthLimit;
                if (BodyMotionData->IsValidIndex(WriteIndex))
                {
                    TArray<double>& TargetFrame = (*BodyMotionData)[WriteIndex];
                    for (int32 i = 0; i < Frame.Num() && i < TargetFrame.Num(); ++i)
                    {

                        TargetFrame[i] = Frame[i]->AsNumber();
                    }
                    *BodyFrameIndex = WriteIndex;
                }
            }
        }
    }
    // ----------------- FaceData -----------------
    const TArray<TSharedPtr<FJsonValue>>* FaceArray = nullptr;
    if (JsonObject->TryGetArrayField(TEXT("face_pred"), FaceArray))
    {
        *FaceFrameIndex = -1;
        int32 FrameCount = FaceArray->Num();
        int32 ElementCount = FrameCount > 0 && (*FaceArray)[0]->Type == EJson::Array
            ? (*FaceArray)[0]->AsArray().Num() : 0;
        UE_LOG(LogTemp, Log, TEXT("FaceData shape: frames=%d, elements per frame=%d"), FrameCount, ElementCount);

        for (auto& FrameValue : *FaceArray)
        {
            if (FrameValue->Type == EJson::Array)
            {
                const TArray<TSharedPtr<FJsonValue>>& Frame = FrameValue->AsArray();
                int32 WriteIndex = (*FaceFrameIndex + 1) % LengthLimit;
                if (FaceMotionData->IsValidIndex(WriteIndex))
                {
                    TArray<double>& TargetFrame = (*FaceMotionData)[WriteIndex];
                    for (int32 i = 0; i < Frame.Num() && i < TargetFrame.Num(); ++i)
                    {
                        TargetFrame[i] = Frame[i]->AsNumber();
                    }
                    *FaceFrameIndex = WriteIndex;

                }
            }
        }


    }
    // ----------------- 整数字段 -----------------
    if (JsonObject->HasField(TEXT("fps")))
    {
        TSharedPtr<FJsonValue> Value = JsonObject->TryGetField(TEXT("fps"));
        if (Value.IsValid())
        {
            if (Value->Type == EJson::Number)
                *fps = FMath::RoundToInt(Value->AsNumber());
            else if (Value->Type == EJson::String)
                *fps = FCString::Atoi(*Value->AsString());

            UE_LOG(LogTemp, Log, TEXT("FPS: %d"), *fps);
        }
    }
    if (JsonObject->HasField(TEXT("frames")))
    {
        TSharedPtr<FJsonValue> Value = JsonObject->TryGetField(TEXT("frames"));
        if (Value.IsValid())
        {
            if (Value->Type == EJson::Number)
                *FrameLength = FMath::RoundToInt(Value->AsNumber());
            else if (Value->Type == EJson::String)
                *FrameLength = FCString::Atoi(*Value->AsString());

            UE_LOG(LogTemp, Log, TEXT("FrameLength: %d"), *FrameLength);
        }
    }
}

void FDataReceiverRunnable::Stop()
{
    // 清理客户端 socket
    if (ClientSocket)
    {
        ClientSocket->Close();
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(ClientSocket);
        ClientSocket = nullptr;
    }
    bRunThread = false;
}

