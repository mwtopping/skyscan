-- name: CreateTransient :one
INSERT INTO transients (id, created_at, updated_at, expstart, exptime, ra1, ra2, dec1, dec2, satnum, imgdata)
VALUES (
	gen_random_uuid(),
	NOW(),
	NOW(),
	$1,
	$2,
	$3,
	$4,
	$5,
	$6,
	$7,
	$8
)
RETURNING *;

-- name: GetSomeTransients :many
SELECT * from transients
ORDER BY RANDOM()
LIMIT $1;

-- name: GetTransient :one
SELECT * from transients
	WHERE id = $1;

-- name: GetAllTransients :many
SELECT * from transients
    ORDER BY satnum, expstart;

-- name: GetTransientsOfSatellite :many
SELECT * from transients
	WHERE satnum = $1;

-- name: Reset :exec
DELETE FROM transients;

-- name: GetUniqueTransients :many
SELECT satnum from transients
GROUP BY satnum;
