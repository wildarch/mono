module {
  %c59774856945_i64 = arith.constant 59774856945 : i64
  %0 = physicalplan.decl_memref memref<1024xi64> at %c59774856945_i64
  %c59774856946_i64 = arith.constant 59774856946 : i64
  %1 = physicalplan.decl_memref memref<1024xi64> at %c59774856946_i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c8 = arith.constant 8 : index
  scf.for %arg0 = %c0 to %c1024 step %c8 {
    %2 = vector.load %0[%arg0] : memref<1024xi64>, vector<8xi64>
    %3 = vector.load %1[%arg0] : memref<1024xi64>, vector<8xi64>
    %c8_i64 = arith.constant 8 : i64
    %c59774856948_i64 = arith.constant 59774856948 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %offset, %errorNoCapacity = physicalplan.claim_slice claim %c8_i64 at %c59774856948_i64 capacity %c1024_i64
    %c59774856947_i64 = arith.constant 59774856947 : i64
    %4 = physicalplan.decl_memref memref<?xi64> at %c59774856947_i64
    vector.store %3, %4[%offset] : memref<?xi64>, vector<8xi64>
  }
}

