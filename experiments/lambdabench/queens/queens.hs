queens :: Int -> Int
queens n = length (qqueens n n)

qqueens :: Int -> Int -> [[Int]]
qqueens k 0 = [[]]
qqueens k n = [(x : xs) | xs <- qqueens k (n -1), x <- [1 .. k], safe x 1 xs]

safe :: Int -> Int -> [Int] -> Bool
safe x d [] = True
safe x d (y : ys) = x /= y && x + d /= y && x - d /= y && safe x (d + 1) ys