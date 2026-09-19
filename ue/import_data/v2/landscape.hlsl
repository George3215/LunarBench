// Port of MI_Landscape0.mdl with its authored USD parameter overrides.
// Soil_N textures are packed RG normal, B specular, A blend height.
struct Helpers {
 float3 decode(float4 p) { float2 xy=p.rg*2-1; return normalize(float3(xy,sqrt(saturate(1-dot(xy,xy))))); }
 float overlay(float m,float h) { return saturate(m>=.5 ? 1-2*(1-m)*(1-h) : 2*m*h); }
 float3 rnm(float3 a,float3 b) { float3 t=float3(a.xy,a.z+1); float3 u=float3(-b.xy,b.z); return t*dot(t,u)-t.z*u; }
};
Helpers H;
float fade=saturate((Depth-1024)/2048);
float2 nearUV=World.xy/400;
float2 farUV=World.xy/1000;
float4 n1=Texture2DSample(N1,N1Sampler,nearUV);
float4 n2=Texture2DSample(N2,N2Sampler,nearUV);
float4 n3=Texture2DSample(N3,N3Sampler,nearUV);
float4 f1=Texture2DSample(N1,N1Sampler,farUV);
float4 f2=Texture2DSample(N2,N2Sampler,farUV);
float4 f3=Texture2DSample(N3,N3Sampler,farUV);
// Original UV0 is the unscaled landscape quad coordinate, not world metres.
float2 uv0=(World.xy-float2(-69694.85770002793,-62437.92710342578))/50;
float4 mask=Texture2DSample(Mask,MaskSampler,uv0*.005);
float b2=H.overlay(mask.g,saturate(pow(max(n2.a,.000001),2)*4));
float b3=H.overlay(mask.r,saturate(pow(max(n3.a,.000001),2)*14));
float4 mix1=lerp(n1,f1,fade),mix2=lerp(n2,f2,fade),mix3=lerp(n3,f3,fade);
float3 norm=lerp(lerp(H.decode(mix1)*float3(1,1,.25),H.decode(mix2)*float3(1,1,.25),b2),H.decode(mix3)*float3(1,1,.25),b3);
norm=lerp(norm,norm*float3(.25,.25,1),fade);
float blend=saturate((Depth-1024)/4096);
float3 df=H.decode(Texture2DSample(Detail,DetailSampler,World.xy/32000))*float3(.075,.075,1);
float3 dn=H.decode(Texture2DSample(Detail,DetailSampler,World.xy/2048))*float3(.1,.1,1);
norm=lerp(norm,H.rnm(norm,df),blend);
NormalOut=normalize(lerp(norm,H.rnm(norm,dn),1-blend));
float3 a1=lerp(Texture2DSample(A1,A1Sampler,nearUV).rgb,Texture2DSample(A1,A1Sampler,farUV).rgb,fade);
float3 a2=lerp(Texture2DSample(A2,A2Sampler,nearUV).rgb,Texture2DSample(A2,A2Sampler,farUV).rgb,fade)*.99;
float3 a3=lerp(Texture2DSample(A3,A3Sampler,nearUV).rgb,Texture2DSample(A3,A3Sampler,farUV).rgb,fade)*.99;
float3 base=lerp(lerp(a1,a2,b2),a3,b3);
float c1=Texture2DSample(Mask,MaskSampler,World.xy/16384).a;
float c2=Texture2DSample(Mask,MaskSampler,World.xy/812900).a;
SpecOut=.5*lerp(lerp(mix1.b,mix2.b,b2),mix3.b,b3);
return saturate(lerp(lerp(base,base*.51,c1),base*.15,c2));
