-- Taken from 'The Lazy Virtual Machine speciﬁcation'
sieve n = last (take n (ssieve [3, 5 ..]))
  where
    ssieve (x : xs) = x : ssieve (filter (noDiv x) xs)
    noDiv x y = (mod x y /= 0)