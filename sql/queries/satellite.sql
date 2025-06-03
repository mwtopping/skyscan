-- name: GetSatellite :one
SELECT * from satellites
	WHERE id = $1;


-- name: ResetSatellites :exec
DELETE FROM satellites;
