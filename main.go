package main

import (
	"database/sql"
	"fmt"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
	"log"
	"net/http"
	"os"
	"skyscan/internal/database"
	"skyscan/internal/satellites"
	"time"
)

func main() {

	interval := time.Duration(3) * time.Hour
	ticker := time.NewTicker(interval)

	go func() {
		for {
			satellites.Retrieve_satellites()
			log.Println("Done retrieving satellites")
			<-ticker.C
		}
	}()

	fmt.Println("Starting SkyScan")

	const port = ":8080"

	godotenv.Load()
	dbURL := os.Getenv("DB_URL")

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatal("Error opening database")
	}

	dbQueries := database.New(db)

	cfg := apiConfig{
		dbQueries: dbQueries,
		Port:      port}

	mux := http.NewServeMux()

	mux.HandleFunc("POST /api/submit/", cfg.handlerRecieveEntry)
	mux.HandleFunc("GET /", cfg.handlerDisplay)
	mux.HandleFunc("GET /transients/{transientID}/", cfg.handlerDisplayOne)
	mux.HandleFunc("GET /satellites/{satelliteID}/", cfg.handlerDisplaySatellite)
	mux.HandleFunc("GET /api/reset/", cfg.handlerReset)

	server := &http.Server{Handler: mux, Addr: cfg.Port}

	log.Printf("Server listening on port%v\n", cfg.Port)
	log.Fatal(server.ListenAndServe())

}

//func handle(w http.ResponseWriter, r *http.Request) {
//	fmt.Println("Got request!")
//}
