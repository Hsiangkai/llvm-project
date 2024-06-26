; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX12,GFX12-SDAG %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX11 %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX12,GFX12-GISEL %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX11 %s

; Scalar data prefetch

define amdgpu_ps void @prefetch_data_sgpr(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_data s[0:1], 0x0, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %ptr, i32 0, i32 0, i32 1)
  ret void
}

define amdgpu_ps void @prefetch_data_sgpr_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr_offset:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_data s[0:1], 0x200, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  %gep = getelementptr float, ptr addrspace(4) %ptr, i32 128
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 1)
  ret void
}

; Check large offsets

define amdgpu_ps void @prefetch_data_sgpr_max_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr_max_offset:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_data s[0:1], 0x7fffff, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_max_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 8388607
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 1)
  ret void
}

define amdgpu_ps void @prefetch_data_sgpr_min_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr_min_offset:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_data s[0:1], -0x800000, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_min_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 -8388608
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 1)
  ret void
}

define amdgpu_ps void @prefetch_data_sgpr_too_large_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-SDAG-LABEL: prefetch_data_sgpr_too_large_offset:
; GFX12-SDAG:       ; %bb.0: ; %entry
; GFX12-SDAG-NEXT:    s_add_nc_u64 s[0:1], s[0:1], 0x800000
; GFX12-SDAG-NEXT:    s_prefetch_data s[0:1], 0x0, null, 0
; GFX12-SDAG-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_too_large_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
;
; GFX12-GISEL-LABEL: prefetch_data_sgpr_too_large_offset:
; GFX12-GISEL:       ; %bb.0: ; %entry
; GFX12-GISEL-NEXT:    s_add_co_u32 s0, s0, 0x800000
; GFX12-GISEL-NEXT:    s_add_co_ci_u32 s1, s1, 0
; GFX12-GISEL-NEXT:    s_prefetch_data s[0:1], 0x0, null, 0
; GFX12-GISEL-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 8388608
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 1)
  ret void
}

; Check divergent address

define amdgpu_ps void @prefetch_data_vgpr(ptr addrspace(1) %ptr) {
; GCN-LABEL: prefetch_data_vgpr:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p1(ptr addrspace(1) %ptr, i32 0, i32 0, i32 1)
  ret void
}

; Check LDS and Scratch, we cannot prefetch it

define amdgpu_ps void @prefetch_data_lds(ptr addrspace(3) inreg %ptr) {
; GCN-LABEL: prefetch_data_lds:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p3(ptr addrspace(3) %ptr, i32 0, i32 0, i32 1)
  ret void
}

define amdgpu_ps void @prefetch_data_scratch(ptr addrspace(5) inreg %ptr) {
; GCN-LABEL: prefetch_data_scratch:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p5(ptr addrspace(5) %ptr, i32 0, i32 0, i32 1)
  ret void
}

; Check supported address spaces

define amdgpu_ps void @prefetch_data_sgpr_flat(ptr inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr_flat:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_data s[0:1], 0x0, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_flat:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.pf(ptr %ptr, i32 0, i32 0, i32 1)
  ret void
}

define amdgpu_ps void @prefetch_data_sgpr_global(ptr addrspace(1) inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr_global:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_data s[0:1], 0x0, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_global:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p1(ptr addrspace(1) %ptr, i32 0, i32 0, i32 1)
  ret void
}

define amdgpu_ps void @prefetch_data_sgpr_constant_32bit(ptr addrspace(6) inreg %ptr) {
; GFX12-LABEL: prefetch_data_sgpr_constant_32bit:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_mov_b32 s1, 0
; GFX12-NEXT:    s_prefetch_data s[0:1], 0x0, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_data_sgpr_constant_32bit:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p6(ptr addrspace(6) %ptr, i32 0, i32 0, i32 1)
  ret void
}

; I$ prefetch

define amdgpu_ps void @prefetch_inst_sgpr(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_inst_sgpr:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_inst s[0:1], 0x0, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_inst_sgpr:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %ptr, i32 0, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @prefetch_inst_sgpr_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_inst_sgpr_offset:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_inst s[0:1], 0x80, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_inst_sgpr_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 128
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 0)
  ret void
}

; Check large offsets

define amdgpu_ps void @prefetch_inst_sgpr_max_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_inst_sgpr_max_offset:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_inst s[0:1], 0x7fffff, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_inst_sgpr_max_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 8388607
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @prefetch_inst_sgpr_min_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-LABEL: prefetch_inst_sgpr_min_offset:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_prefetch_inst s[0:1], -0x800000, null, 0
; GFX12-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_inst_sgpr_min_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 -8388608
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 0)
  ret void
}

define amdgpu_ps void @prefetch_inst_sgpr_too_large_offset(ptr addrspace(4) inreg %ptr) {
; GFX12-SDAG-LABEL: prefetch_inst_sgpr_too_large_offset:
; GFX12-SDAG:       ; %bb.0: ; %entry
; GFX12-SDAG-NEXT:    s_add_nc_u64 s[0:1], s[0:1], 0x800000
; GFX12-SDAG-NEXT:    s_prefetch_inst s[0:1], 0x0, null, 0
; GFX12-SDAG-NEXT:    s_endpgm
;
; GFX11-LABEL: prefetch_inst_sgpr_too_large_offset:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_endpgm
;
; GFX12-GISEL-LABEL: prefetch_inst_sgpr_too_large_offset:
; GFX12-GISEL:       ; %bb.0: ; %entry
; GFX12-GISEL-NEXT:    s_add_co_u32 s0, s0, 0x800000
; GFX12-GISEL-NEXT:    s_add_co_ci_u32 s1, s1, 0
; GFX12-GISEL-NEXT:    s_prefetch_inst s[0:1], 0x0, null, 0
; GFX12-GISEL-NEXT:    s_endpgm
entry:
  %gep = getelementptr i8, ptr addrspace(4) %ptr, i32 8388608
  tail call void @llvm.prefetch.p4(ptr addrspace(4) %gep, i32 0, i32 0, i32 0)
  ret void
}

declare void @llvm.prefetch.pf(ptr nocapture readonly, i32, i32, i32)
declare void @llvm.prefetch.p1(ptr addrspace(1) nocapture readonly, i32, i32, i32)
declare void @llvm.prefetch.p3(ptr addrspace(3) nocapture readonly, i32, i32, i32)
declare void @llvm.prefetch.p4(ptr addrspace(4) nocapture readonly, i32, i32, i32)
declare void @llvm.prefetch.p5(ptr addrspace(5) nocapture readonly, i32, i32, i32)
declare void @llvm.prefetch.p6(ptr addrspace(6) nocapture readonly, i32, i32, i32)
