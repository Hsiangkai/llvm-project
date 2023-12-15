//=- VectorEmulateMaskedLoadStore.cpp - Emulate 'vector.maskedload/store' op =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to emulate the
// 'vector.maskedload' and 'vector.maskedstore' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"

using namespace mlir;

namespace {

std::optional<int64_t> getMaskLeadingOneLength(Value mask) {
  if (auto m = mask.getDefiningOp<vector::ConstantMaskOp>()) {
    ArrayAttr masks = m.getMaskDimSizes();
    assert(masks.size() == 1 && "Only support 1-D mask.");
    return llvm::cast<IntegerAttr>(masks[0]).getInt();
  } else if (auto m = mask.getDefiningOp<vector::CreateMaskOp>()) {
    auto maskOperands = m.getOperands();
    assert(maskOperands.size() == 1 && "Only support 1-D mask.");
    if (auto constantOp = maskOperands[0].getDefiningOp<arith::ConstantOp>()) {
      return llvm::cast<IntegerAttr>(constantOp.getValue()).getInt();
    }
  }
  return {};
}

/// Convert vector.maskedload
struct VectorMaskedLoadOpConverter final
    : OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp maskedLoadOp,
                                PatternRewriter &rewriter) const override {
    VectorType maskVType = maskedLoadOp.getMaskVectorType();
    if (maskVType.getShape().size() != 1)
      return rewriter.notifyMatchFailure(
          maskedLoadOp, "expected vector.maskedstore with 1-D mask");

    Location loc = maskedLoadOp.getLoc();
    std::optional<int64_t> ones =
        getMaskLeadingOneLength(maskedLoadOp.getMask());
    if (ones) {
      /// Converted to:
      ///
      ///   %partial = vector.load %base[%idx_0, %idx_1]
      ///   %value = vector.insert_strided_slice %partial, %pass_thru
      ///            {offsets = [0], strides = [1]}
      Type vectorType =
          VectorType::get(*ones, maskedLoadOp.getMemRefType().getElementType());
      auto loadOp = rewriter.create<vector::LoadOp>(
          loc, vectorType, maskedLoadOp.getBase(), maskedLoadOp.getIndices());
      auto insertedValue = rewriter.create<vector::InsertStridedSliceOp>(
          loc, loadOp, maskedLoadOp.getPassThru(),
          /*offsets=*/ArrayRef<int64_t>({0}),
          /*strides=*/ArrayRef<int64_t>({1}));
      rewriter.replaceOp(maskedLoadOp, insertedValue);
    } else {
      /// Converted to:
      ///
      ///   %ivalue = %pass_thru
      ///   %m = vector.extract %mask[0]
      ///   %result0 = scf.if %m {
      ///     %v = memref.load %base[%idx_0, %idx_1]
      ///     %combined = vector.insert %v, %ivalue[0]
      ///     scf.yield %combined
      ///   } else {
      ///     scf.yield %ivalue
      ///   }
      ///   %m = vector.extract %mask[1]
      ///   %result1 = scf.if %m {
      ///     %v = memref.load %base[%idx_0, %idx_1 + 1]
      ///     %combined = vector.insert %v, %result0[1]
      ///     scf.yield %combined
      ///   } else {
      ///     scf.yield %result0
      ///   }
      ///   ...
      ///
      int64_t maskLength = maskVType.getShape()[0];

      Type indexType = rewriter.getIndexType();
      Value mask = maskedLoadOp.getMask();
      Value base = maskedLoadOp.getBase();
      Value iValue = maskedLoadOp.getPassThru();
      auto indices = llvm::to_vector_of<Value>(maskedLoadOp.getIndices());
      Value one = rewriter.create<arith::ConstantOp>(
          loc, indexType, IntegerAttr::get(indexType, 1));
      for (int64_t i = 0; i < maskLength; ++i) {
        auto maskBit = rewriter.create<vector::ExtractOp>(loc, mask, i);

        auto ifOp = rewriter.create<scf::IfOp>(
            loc, maskBit,
            [&](OpBuilder &builder, Location loc) {
              auto loadedValue =
                  builder.create<memref::LoadOp>(loc, base, indices);
              auto combinedValue =
                  builder.create<vector::InsertOp>(loc, loadedValue, iValue, i);
              builder.create<scf::YieldOp>(loc, combinedValue.getResult());
            },
            [&](OpBuilder &builder, Location loc) {
              builder.create<scf::YieldOp>(loc, iValue);
            });
        iValue = ifOp.getResult(0);

        indices.back() =
            rewriter.create<arith::AddIOp>(loc, indices.back(), one);
      }

      rewriter.replaceOp(maskedLoadOp, iValue);
    }

    return success();
  }
};

/// Convert vector.maskedstore
struct VectorMaskedStoreOpConverter final
    : OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp maskedStoreOp,
                                PatternRewriter &rewriter) const override {
    VectorType maskVType = maskedStoreOp.getMaskVectorType();
    if (maskVType.getShape().size() != 1)
      return rewriter.notifyMatchFailure(
          maskedStoreOp, "expected vector.maskedstore with 1-D mask");

    Location loc = maskedStoreOp.getLoc();
    std::optional<int64_t> ones =
        getMaskLeadingOneLength(maskedStoreOp.getMask());
    if (ones) {
      /// Converted to:
      ///
      ///   %partial = vector.extract_strided_slice %value
      ///              {offsets = [0], sizes = [1], strides = [1]}
      ///   vector.store %partial, %base[%idx_0, %idx_1]
      Value value = maskedStoreOp.getValueToStore();
      Value extractedValue = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, value, /*offsets=*/ArrayRef<int64_t>({0}),
          /*sizes=*/ArrayRef<int64_t>(*ones),
          /*strides=*/ArrayRef<int64_t>({1}));
      auto storeOp = rewriter.create<vector::StoreOp>(
          loc, extractedValue, maskedStoreOp.getBase(),
          maskedStoreOp.getIndices());
      rewriter.replaceOp(maskedStoreOp, storeOp);
    } else {
      /// Converted to:
      ///
      ///   %m = vector.extract %mask[0]
      ///   scf.if %m {
      ///     %extracted = vector.extract %value[0]
      ///     memref.store %extracted, %base[%idx_0, %idx_1]
      ///   }
      ///   %m = vector.extract %mask[1]
      ///   scf.if %m {
      ///     %extracted = vector.extract %value[1]
      ///     memref.store %extracted, %base[%idx_0, %idx_1 + 1]
      ///   }
      ///   ...
      int64_t maskLength = maskVType.getShape()[0];

      Type indexType = rewriter.getIndexType();
      Value mask = maskedStoreOp.getMask();
      Value base = maskedStoreOp.getBase();
      Value value = maskedStoreOp.getValueToStore();
      auto indices = llvm::to_vector_of<Value>(maskedStoreOp.getIndices());
      Value one = rewriter.create<arith::ConstantOp>(
          loc, indexType, IntegerAttr::get(indexType, 1));
      for (int64_t i = 0; i < maskLength; ++i) {
        auto maskBit = rewriter.create<vector::ExtractOp>(loc, mask, i);

        auto ifOp = rewriter.create<scf::IfOp>(loc, maskBit, /*else=*/false);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto extractedValue = rewriter.create<vector::ExtractOp>(loc, value, i);
        rewriter.create<memref::StoreOp>(loc, extractedValue, base, indices);

        rewriter.setInsertionPointAfter(ifOp);
        indices.back() =
            rewriter.create<arith::AddIOp>(loc, indices.back(), one);
      }

      rewriter.eraseOp(maskedStoreOp);
    }

    return success();
  }
};

} // namespace

void mlir::vector::populateVectorMaskedLoadStoreEmulationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<VectorMaskedLoadOpConverter, VectorMaskedStoreOpConverter>(
      patterns.getContext(), benefit);
}
