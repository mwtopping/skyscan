-- +goose Up
CREATE TABLE satellites (
	id int PRIMARY KEY,
	name TEXT NOT NULL,
	created_at TIMESTAMP NOT NULL,
	updated_at TIMESTAMP NOT NULL,
	epoch TIMESTAMP NOT NULL,
	line1 TEXT,
	line2 TEXT
);

-- +goose Down
DROP TABLE satellites;
