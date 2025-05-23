-- +goose Up
CREATE TABLE transients (
	id UUID PRIMARY KEY,
	created_at TIMESTAMP NOT NULL,
	updated_at TIMESTAMP NOT NULL,
	expstart TIMESTAMP NOT NULL,
	exptime float NOT NULL,
	RA1 float NOT NULL,
	RA2 float NOT NULL,
	DEC1 float NOT NULL,
	DEC2 float NOT NULL,
	SATNUM int,
	IMGDATA TEXT NOT NULL
);


-- +goose Down
DROP TABLE transients;
