-- name: CreateElement :one
INSERT INTO ELEMENTS (
	id, satnum, name, created_at, updated_at, epoch, line1, line2
	)
VALUES (
	gen_random_uuid(),
	$1,
	$2,
	NOW(),
	NOW(),
	$3,
	$4,
	$5)
RETURNING *;


-- name: ResetElements :exec
DELETE FROM elements;

-- name: GetElementWithEpoch :many
SELECT id, satnum, epoch from ELEMENTS
WHERE satnum=$1 AND epoch=$2;
