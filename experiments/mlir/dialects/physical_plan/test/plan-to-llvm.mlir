  func.func @test() {
    %c0 = arith.constant 0 : i64
    %c1024 = arith.constant 1024 : i64
    %c8 = arith.constant 8 : i64
    scf.for %arg0 = %c0 to %c1024 step %c8 : i64 {
    }

    return
  }