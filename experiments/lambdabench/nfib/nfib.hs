-- Taken from 'The Lazy Virtual Machine speciﬁcation'
nfib :: Int -> Int
nfib 0 = 1
nfib 1 = 1
nfib n = 1 + nfib (n -1) + nfib (n -2)