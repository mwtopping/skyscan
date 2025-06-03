-- +goose Up
CREATE TABLE elements (
	id UUID PRIMARY KEY,
	satnum int NOT NULL,
	name TEXT NOT NULL,
	created_at TIMESTAMP NOT NULL,
	updated_at TIMESTAMP NOT NULL,
	epoch TIMESTAMP NOT NULL,
	line1 TEXT NOT NULL,
	line2 TEXT NOT NULL
);

-- +goose Down
DROP TABLE satellites;
