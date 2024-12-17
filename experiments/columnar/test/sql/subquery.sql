SELECT *
FROM part
WHERE p_partkey = (
    SELECT ps_partkey
    FROM partsupp
    WHERE ps_partkey = p_partkey
);