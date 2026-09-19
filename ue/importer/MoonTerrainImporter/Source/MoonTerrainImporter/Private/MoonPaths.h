#pragma once
#include "HAL/PlatformMisc.h"
#include "Misc/Paths.h"

// Launcher passes deployment paths; defaults preserve the original local demo.
namespace MoonPaths
{
inline FString Root()
{
    FString Value=FPlatformMisc::GetEnvironmentVariable(TEXT("LUNARBENCH_WORKSPACE"));
    return Value.IsEmpty()?TEXT("/home/lry/MoonUnrealEnv/MoonSim"):Value;
}
inline FString File(const TCHAR* Relative){return FPaths::Combine(Root(),Relative);}
inline FString Python()
{
    FString Value=FPlatformMisc::GetEnvironmentVariable(TEXT("LUNARBENCH_PYTHON"));
    return Value.IsEmpty()?TEXT("/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python"):Value;
}
// 该地图是否应该由本插件接管（输入处理器 + Python 接收器 + 物理子进程）。
// 旧 demo 地图名永远可用；LUNARBENCH_UE_MAP 指定的地图同样接管。
// 这是个超集判断——换皮地图是原图的副本，接管逻辑完全相同，只是名字不同。
inline bool IsGo2Map(const FString& WorldName)
{
    if(WorldName==TEXT("MoonTerrain_Go2_FullRender"))return true;
    const FString Value=FPlatformMisc::GetEnvironmentVariable(TEXT("LUNARBENCH_UE_MAP"));
    return !Value.IsEmpty() && WorldName==FPaths::GetBaseFilename(Value);
}
inline bool TaskMode(){return !FPlatformMisc::GetEnvironmentVariable(TEXT("MOONSIM_TASK_DIR")).IsEmpty();}
inline bool ExternalPhysics(){return FPlatformMisc::GetEnvironmentVariable(TEXT("LUNARBENCH_EXTERNAL_PHYSICS"))==TEXT("1");}
}
