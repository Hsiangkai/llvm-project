; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt < %s -passes=newgvn -S | FileCheck %s

@last = external global [65 x ptr]

define i32 @NextRootMove(i32 %wtm, i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: define i32 @NextRootMove
; CHECK-SAME: (i32 [[WTM:%.*]], i32 [[X:%.*]], i32 [[Y:%.*]], i32 [[Z:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca ptr, align 8
; CHECK-NEXT:    [[TMP17618:%.*]] = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
; CHECK-NEXT:    store ptr [[TMP17618]], ptr [[A]], align 8
; CHECK-NEXT:    br label [[COND_TRUE116:%.*]]
; CHECK:       cond_true116:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X]], [[Y]]
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_TRUE128:%.*]], label [[COND_TRUE145:%.*]]
; CHECK:       cond_true128:
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i32 [[X]], [[Z]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[BB98_BACKEDGE:%.*]], label [[RETURN_LOOPEXIT:%.*]]
; CHECK:       bb98.backedge:
; CHECK-NEXT:    br label [[COND_TRUE116]]
; CHECK:       cond_true145:
; CHECK-NEXT:    br i1 false, label [[BB98_BACKEDGE]], label [[RETURN_LOOPEXIT]]
; CHECK:       return.loopexit:
; CHECK-NEXT:    br label [[RETURN:%.*]]
; CHECK:       return:
; CHECK-NEXT:    ret i32 0
;
entry:
  %A = alloca ptr
  %tmp17618 = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
  store ptr %tmp17618, ptr %A
  br label %cond_true116

cond_true116:
  %cmp = icmp eq i32 %x, %y
  br i1 %cmp, label %cond_true128, label %cond_true145

cond_true128:
  %tmp17625 = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
  store ptr %tmp17625, ptr %A
  %cmp1 = icmp eq i32 %x, %z
  br i1 %cmp1 , label %bb98.backedge, label %return.loopexit

bb98.backedge:
  br label %cond_true116

cond_true145:
  %tmp17631 = load ptr, ptr getelementptr ([65 x ptr], ptr @last, i32 0, i32 1), align 4
  store ptr %tmp17631, ptr %A
  br i1 false, label %bb98.backedge, label %return.loopexit

return.loopexit:
  br label %return

return:
  ret i32 0
}
