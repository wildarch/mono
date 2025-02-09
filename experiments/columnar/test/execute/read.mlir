llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @col_table_column_open(i64, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
llvm.func @col_print_write(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
llvm.func @col_print_open() -> !llvm.ptr attributes {sym_visibility = "private"}
llvm.func @col_table_scanner_claim_chunk(!llvm.ptr) -> !llvm.struct<(i64, i64)> attributes {sym_visibility = "private"}
llvm.func @col_table_column_read(!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
llvm.func @col_table_scanner_open(i64) -> !llvm.ptr attributes {sym_visibility = "private"}
llvm.func @col_print_chunk_append(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) attributes {sym_visibility = "private"}
llvm.func @col_print_chunk_alloc(i64) -> !llvm.ptr attributes {sym_visibility = "private"}
llvm.func @pipe0_globalOpen() -> !llvm.ptr attributes { llvm.emit_c_interface, sym_visibility = "public" } {
  %0 = llvm.mlir.constant(3 : index) : i64
  %1 = llvm.mlir.constant(42 : index) : i64
  %2 = llvm.call @col_table_scanner_open(%1) : (i64) -> !llvm.ptr
  %3 = llvm.call @col_table_column_open(%1, %0) : (i64, i64) -> !llvm.ptr
  %4 = llvm.call @col_print_open() : () -> !llvm.ptr
  %5 = llvm.mlir.zero : !llvm.ptr
  %6 = llvm.getelementptr %5[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, ptr)>
  %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
  %8 = llvm.call @malloc(%7) : (i64) -> !llvm.ptr
  %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
  %10 = llvm.insertvalue %2, %9[0] : !llvm.struct<(ptr, ptr, ptr)> 
  %11 = llvm.insertvalue %3, %10[1] : !llvm.struct<(ptr, ptr, ptr)> 
  %12 = llvm.insertvalue %4, %11[2] : !llvm.struct<(ptr, ptr, ptr)> 
  llvm.store %12, %8 : !llvm.struct<(ptr, ptr, ptr)>, !llvm.ptr
  llvm.return %8 : !llvm.ptr
}
llvm.func @pipe0_body(%arg0: !llvm.ptr) -> i1 {
  %0 = llvm.mlir.constant(1 : index) : i64
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, ptr)>
  %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, ptr)> 
  %4 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, ptr)>
  %5 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, ptr)> 
  %6 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, ptr)>
  %7 = llvm.extractvalue %6[2] : !llvm.struct<(ptr, ptr, ptr)> 
  %8 = llvm.call @col_table_scanner_claim_chunk(%3) : (!llvm.ptr) -> !llvm.struct<(i64, i64)>
  %9 = llvm.extractvalue %8[0] : !llvm.struct<(i64, i64)> 
  %10 = llvm.extractvalue %8[1] : !llvm.struct<(i64, i64)> 
  %11 = llvm.icmp "ugt" %10, %1 : i64
  %12 = llvm.mlir.constant(1 : index) : i64
  %13 = llvm.mlir.zero : !llvm.ptr
  %14 = llvm.getelementptr %13[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
  %16 = llvm.mlir.constant(64 : index) : i64
  %17 = llvm.add %15, %16 : i64
  %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
  %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
  %20 = llvm.mlir.constant(1 : index) : i64
  %21 = llvm.sub %16, %20 : i64
  %22 = llvm.add %19, %21 : i64
  %23 = llvm.urem %22, %16 : i64
  %24 = llvm.sub %22, %23 : i64
  %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
  %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %27 = llvm.insertvalue %18, %26[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %29 = llvm.mlir.constant(0 : index) : i64
  %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %31 = llvm.insertvalue %10, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %32 = llvm.insertvalue %12, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  llvm.br ^bb1(%1 : i64)
^bb1(%33: i64):  // 2 preds: ^bb0, ^bb2
  %34 = llvm.icmp "slt" %33, %10 : i64
  llvm.cond_br %34, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %35 = llvm.extractvalue %32[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %36 = llvm.getelementptr %35[%33] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %33, %36 : i64, !llvm.ptr
  %37 = llvm.add %33, %0 : i64
  llvm.br ^bb1(%37 : i64)
^bb3:  // pred: ^bb1
  %38 = llvm.mlir.constant(1 : index) : i64
  %39 = llvm.mlir.zero : !llvm.ptr
  %40 = llvm.getelementptr %39[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
  %42 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
  %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %46 = llvm.mlir.constant(0 : index) : i64
  %47 = llvm.insertvalue %46, %45[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %48 = llvm.insertvalue %10, %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %49 = llvm.insertvalue %38, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %50 = llvm.extractvalue %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %51 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %52 = llvm.extractvalue %49[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %53 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %54 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  llvm.call @col_table_column_read(%5, %9, %10, %50, %51, %52, %53, %54) : (!llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
  %55 = llvm.call @col_print_chunk_alloc(%10) : (i64) -> !llvm.ptr
  %56 = llvm.extractvalue %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %57 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %58 = llvm.extractvalue %49[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %59 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %60 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %61 = llvm.extractvalue %32[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %62 = llvm.extractvalue %32[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %63 = llvm.extractvalue %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %64 = llvm.extractvalue %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %65 = llvm.extractvalue %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  llvm.call @col_print_chunk_append(%55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
  llvm.call @col_print_write(%7, %55) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return %11 : i1
}
llvm.func @pipe0_globalClose(%arg0: !llvm.ptr) {
  llvm.return
}
columnar.pipeline_ref global_open @pipe0_globalOpen body @pipe0_body global_close @pipe0_globalClose