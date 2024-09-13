  func.func @test() {
    %c59774856945_i64 = arith.constant 59774856945 : i64
    %0 = physicalplan.decl_memref memref<1024xi64> at %c59774856945_i64
    %c59774856946_i64 = arith.constant 59774856946 : i64
    %1 = physicalplan.decl_memref memref<1024xi64> at %c59774856946_i64
    %c0 = arith.constant 0 : i64
    %c1024 = arith.constant 1024 : i64
    %c8 = arith.constant 8 : i64
    scf.for %arg0 = %c0 to %c1024 step %c8 : i64 {
    }

    return
  }