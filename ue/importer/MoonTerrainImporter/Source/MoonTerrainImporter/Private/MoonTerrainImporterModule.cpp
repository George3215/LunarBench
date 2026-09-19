#include "Modules/ModuleManager.h"
#include "MoonPaths.h"
#include "MoonViewportLibrary.h"
#include "Editor/EditorPerformanceSettings.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/Application/IInputProcessor.h"
#include "Editor.h"
#include "UnrealClient.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"
#include "HAL/PlatformProcess.h"
#include "IPythonScriptPlugin.h"
#include "GameFramework/Actor.h"
#include "Debug/DebugDrawService.h"
#include "Engine/Canvas.h"
#include "Engine/Engine.h"
#include "EditorViewportClient.h"
#include "SEditorViewport.h"
#include "Slate/SceneViewport.h"
#include "Widgets/SViewport.h"
#include "Misc/CommandLine.h"
#include "Misc/CoreDelegates.h"

// Only the dedicated patch viewport captures controls. Text fields and other maps
// are never intercepted; RMB keeps normal editor camera navigation available.
class FMoonGo2Input : public IInputProcessor
{
    FSocket* Socket=nullptr;
    TSharedPtr<FInternetAddr> Address;
    FProcHandle Physics;
    TWeakObjectPtr<UWorld> World;
    bool HadFocus=false;
    double Heartbeat=0;
    int Gear=2;
    FDelegateHandle DrawHandle;
    double ReadyTime=0;
    int SmokeStage=0;
    bool PreviousThrottle=true;
    bool ThrottleOverridden=false;
public:
    FMoonGo2Input()
    {
        Socket=ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateSocket(NAME_DGram,TEXT("Go2 local input"),false);
        Address=ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();bool Valid=false;Address->SetIp(TEXT("127.0.0.1"),Valid);Address->SetPort(19401);
        DrawHandle=UDebugDrawService::Register(TEXT("OnScreenDebug"),FDebugDrawDelegate::CreateRaw(this,&FMoonGo2Input::Draw));
    }
    ~FMoonGo2Input()
    {
        UMoonViewportLibrary::ReleaseCamera();
        if(ThrottleOverridden)GetMutableDefault<UEditorPerformanceSettings>()->bThrottleCPUWhenNotForeground=PreviousThrottle;
        UDebugDrawService::Unregister(DrawHandle);
        if(Physics.IsValid()){FPlatformProcess::TerminateProc(Physics);FPlatformProcess::CloseProc(Physics);}
        if(Socket){Socket->Close();ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);}
    }
    void Send(char Key){if(MoonPaths::ExternalPhysics() && !MoonPaths::TaskMode())return;int32 Sent=0;if(Socket)Socket->SendTo(reinterpret_cast<uint8*>(&Key),1,Sent,*Address);}
    void Draw(UCanvas* Canvas,APlayerController*)
    {
        if(!World.IsValid() || !Canvas || !GEngine || (MoonPaths::ExternalPhysics() && !MoonPaths::TaskMode()))return;
        const TCHAR* Names[]={TEXT("SLOW 0.15 m/s"),TEXT("MEDIUM 0.30 m/s"),TEXT("FAST 0.50 m/s")};
        Canvas->SetDrawColor(FColor::Yellow);
        Canvas->DrawText(GEngine->GetMediumFont(),FString::Printf(TEXT("GO2  |  Gear %d: %s  |  H: slower / L: faster"),Gear,Names[Gear-1]),16,16);
        Canvas->DrawText(GEngine->GetSmallFont(),TEXT("Click viewport. W/S/A/D: 0.5m step | Q/E: 15deg | SPACE: stop | R: reset | P: pause | RMB: camera"),16,44);
    }
    bool Focused() const
    {
        return World.IsValid() && GEditor && GEditor->GetActiveViewport() && GEditor->GetActiveViewport()->HasFocus()
            && !FSlateApplication::Get().GetPressedMouseButtons().Contains(EKeys::RightMouseButton);
    }
    void Tick(float, FSlateApplication&, TSharedRef<ICursor>) override
    {
        if(!GEditor || IsRunningCommandlet())return;
        UWorld* Current=GEditor->GetEditorWorldContext().World();
        bool Active=Current && MoonPaths::IsGo2Map(Current->GetName());
        if(Active && World.Get()!=Current)
        {
            int Count=0;for(TActorIterator<AActor> It(Current);It;++It)if(It->ActorHasTag(TEXT("Go2Visual")))Count++;
            if(Count!=33 || !IPythonScriptPlugin::Get()->IsPythonAvailable())return;
            World=Current;
            if(!ThrottleOverridden){auto* Settings=GetMutableDefault<UEditorPerformanceSettings>();PreviousThrottle=Settings->bThrottleCPUWhenNotForeground;Settings->bThrottleCPUWhenNotForeground=false;ThrottleOverridden=true;}
            Gear=2;ReadyTime=FPlatformTime::Seconds();
            for(FEditorViewportClient* C:GEditor->GetAllViewportClients())if(C->GetWorld()==Current){C->EngineShowFlags.SetOnScreenDebug(true);C->EngineShowFlags.SetGrid(false);C->ExposureSettings.bFixed=true;C->ExposureSettings.FixedEV100=1;C->SetRealtime(true);}
            IPythonScriptPlugin::Get()->ExecPythonCommand(*MoonPaths::File(TEXT("mujoco/ue_bridge.py")));
            if(!MoonPaths::ExternalPhysics())
            {
                FString Args=FString::Printf(TEXT("\"%s\" --no-viewer --ue"),*MoonPaths::File(TEXT("mujoco/run.py")));
                if(FPlatformMisc::GetEnvironmentVariable(TEXT("LUNARBENCH_MUJOCO_VIEWER"))==TEXT("1"))Args+=TEXT(" --viewer");
                Physics=FPlatformProcess::CreateProc(*MoonPaths::Python(),*Args,false,true,true,nullptr,0,*MoonPaths::Root(),nullptr);
            }
            UE_LOG(LogTemp,Display,TEXT("GO2_VIEWPORT_CONTROL_READY W/S/A/D=0.5m Q/E=15deg H/L=gear Space=stop"));
        }
        if(!Active && World.IsValid())
        {
            Send(' ');UMoonViewportLibrary::ReleaseCamera();World.Reset();
            if(ThrottleOverridden){GetMutableDefault<UEditorPerformanceSettings>()->bThrottleCPUWhenNotForeground=PreviousThrottle;ThrottleOverridden=false;}
            if(Physics.IsValid()){FPlatformProcess::TerminateProc(Physics);FPlatformProcess::CloseProc(Physics);Physics.Reset();}
        }
        bool Focus=Active && Focused();
        if(HadFocus && !Focus)Send(' ');
        HadFocus=Focus;
        double Now=FPlatformTime::Seconds();
        if(Focus && Now-Heartbeat>.1){Send('_');Heartbeat=Now;}
        if(Active && FParse::Param(FCommandLine::Get(),TEXT("Go2InputSmoke")))
        {
            if(SmokeStage==0 && Now-ReadyTime>10)
            {
                auto* Client=static_cast<FEditorViewportClient*>(GEditor->GetActiveViewport()->GetClient());
                auto Widget=Client->GetEditorViewportWidget()->GetSceneViewport()->GetViewportWidget().Pin();
                FSlateApplication::Get().SetKeyboardFocus(Widget,EFocusCause::SetDirectly);
                auto Key=[&](FKey K,bool Repeat=false){return FSlateApplication::Get().ProcessKeyDownEvent(FKeyEvent(K,FModifierKeysState(),0,Repeat,0,0));};
                check(Focused());check(Key(EKeys::H));check(Gear==1);check(Key(EKeys::L));check(Gear==2);check(Key(EKeys::L));check(Gear==3);check(Key(EKeys::H));check(Gear==2);check(Key(EKeys::H,true));check(Gear==2);
                check(Key(EKeys::W));check(Key(EKeys::W,true));
                UE_LOG(LogTemp,Display,TEXT("GO2_NATIVE_INPUT_SMOKE_PASS focus=viewport gear_limits=pass autorepeat=ignored displacement=W_once"));
                SmokeStage=1;
            }
            else if(SmokeStage==1 && Now-ReadyTime>18){Send(' ');Send('r');SmokeStage=2;}
        }
    }
    bool HandleKeyDownEvent(FSlateApplication&,const FKeyEvent& Event) override
    {
        if((MoonPaths::ExternalPhysics() && !MoonPaths::TaskMode()) || !Focused() || Event.IsControlDown() || Event.IsAltDown() || Event.IsCommandDown())return false;
        const FString Name=Event.GetKey().GetFName().ToString();
        char Key=0;
        if(Name==TEXT("SpaceBar"))Key=' ';
        else if(Name.Len()==1){TCHAR C=FChar::ToLower(Name[0]);if(FString(TEXT("wasdqehlrpb")).Contains(FString::Chr(C)))Key=char(C);}
        if(!Key)return false;
        if(!Event.IsRepeat()){if(Key=='h')Gear=FMath::Max(1,Gear-1);if(Key=='l')Gear=FMath::Min(3,Gear+1);Send(Key);UE_LOG(LogTemp,Display,TEXT("GO2_VIEWPORT_KEY %c gear=%d"),Key,Gear);}
        return true;
    }
};

class FMoonTerrainModule : public IModuleInterface
{
    TSharedPtr<FMoonGo2Input> Input;
    FDelegateHandle PreExit;
    void StopInput()
    {
        if(MoonPaths::TaskMode() && IPythonScriptPlugin::Get()->IsPythonAvailable())
            IPythonScriptPlugin::Get()->ExecPythonCommand(TEXT("import unreal; f=getattr(unreal, '_moon_task_cleanup', None); f() if f else None"));
        if(Input && FSlateApplication::IsInitialized())FSlateApplication::Get().UnregisterInputPreProcessor(Input);
        Input.Reset();
    }
    void StartupModule() override
    {
        if(!IsRunningCommandlet() && FSlateApplication::IsInitialized())
        {Input=MakeShared<FMoonGo2Input>();FSlateApplication::Get().RegisterInputPreProcessor(Input,0);}
        // Release UObject-backed camera/settings before the object system shuts down.
        PreExit=FCoreDelegates::OnEnginePreExit.AddRaw(this,&FMoonTerrainModule::StopInput);
    }
    void ShutdownModule() override
    {
        FCoreDelegates::OnEnginePreExit.Remove(PreExit);
        StopInput();
    }
};

IMPLEMENT_MODULE(FMoonTerrainModule, MoonTerrainImporter)
