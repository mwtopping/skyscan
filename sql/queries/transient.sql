-- name: CreateTransient :one
INSERT INTO transients (id, created_at, updated_at, expstart, exptime, ra1, ra2, dec1, dec2)
VALUES (
	gen_random_uuid(),
	NOW(),
	NOW(),
	$1,
	$2,
	$3,
	$4,
	$5,
	$6
)
RETURNING *;

-- name: GetSomeTransients :many
SELECT * from transients
ORDER BY RANDOM()
LIMIT $1;

