package main

import (
	"skyscan/internal/database"
)

type apiConfig struct {
	Port      string
	dbQueries *database.Queries
}
