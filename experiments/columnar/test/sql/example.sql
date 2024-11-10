SELECT name, SUM(tt.hours)
FROM Employee e, TimeTransaction tt
WHERE e.id = tt.employee_id
GROUP BY name;