package satellites

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"skyscan/internal/database"
	"strconv"
	"time"
)

func Retrieve_satellites() {
	log.Println("Retrieving satelite data from celestrak")

	tle_reader := get_current_tles()

	parsed_tles := parse_tle_bytes(tle_reader)

	godotenv.Load()
	dbURL := os.Getenv("DB_URL")

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatal("Error opening database")
	}

	dbQueries := database.New(db)

	Nupdates := 0

	for _, val := range parsed_tles {
		queryparams := database.GetElementWithEpochParams{Satnum: val.Satnum, Epoch: val.Epoch}
		existingElements, err := dbQueries.GetElementWithEpoch(context.Background(), queryparams)
		if len(existingElements) != 0 {
		} else {
			_, err = dbQueries.CreateElement(context.Background(), val)
			if err != nil {
				fmt.Println("Error inserting element into database", err)
			} else {
				Nupdates += 1
			}
		}
	}
	log.Printf("Updated %d entries\n", Nupdates)
}

func get_current_tles() io.Reader {
	currentUTC := time.Now().UTC()

	formattedTime := fmt.Sprintf("TLEs_%d-%02d-%02d-%02d.txt", currentUTC.Year(),
		currentUTC.Month(),
		currentUTC.Day(),
		currentUTC.Hour())

	filepath := filepath.Join("/Users/michael/workspace/github.com/mwtopping/skyscan/internal/satellites/elements/", formattedTime)

	//check if file exists
	_, err := os.Stat(filepath)
	if errors.Is(err, os.ErrNotExist) {
		fmt.Println("Current files out of date, pulling from celestrak.")
		res, err := http.Get("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle")
		//res, err := http.Get("https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle")
		defer res.Body.Close()

		if err != nil {
			fmt.Println(err)
		}

		res_bytes, err := io.ReadAll(res.Body)
		if err != nil {
			fmt.Println("Error parsing response body", err)
		}

		err = os.WriteFile(filepath, res_bytes, 0644)
		if err != nil {
			fmt.Println("Error writing File", err)
		}
		fmt.Println(res.Body)
		return bytes.NewReader(res_bytes)
	} else {
		fmt.Println("Already have file, reading from disk.")
		f, err := os.Open(filepath)
		if err != nil {
			fmt.Println("Error opening file: ", err)
			return nil
		}
		return f
	}

	// this should never be reached, but the LSP is dumb
	return nil
}

func parse_tle_bytes(r io.Reader) map[int]database.CreateElementParams {
	fmt.Println("Parsing Bytes")
	parsed_tles := make(map[int]database.CreateElementParams)

	scanner := bufio.NewScanner(r)

	for {
		newElement := database.CreateElementParams{}
		scanner.Scan()
		name := scanner.Text()
		scanner.Scan()
		line1 := scanner.Text()
		scanner.Scan()
		line2 := scanner.Text()
		if len(scanner.Text()) == 0 {
			break
		}
		//		fmt.Println(scanner.Text())

		id, epoch := get_element_info(line1)
		newElement.Satnum = int32(id)
		newElement.Epoch = epoch
		newElement.Name = name
		newElement.Line1 = line1
		newElement.Line2 = line2

		parsed_tles[id] = newElement
	}

	return parsed_tles
}

func get_element_info(s string) (int, time.Time) {
	satnum, err := strconv.Atoi(s[2:7])

	year, err := strconv.Atoi(s[18:20])
	if err != nil {
		fmt.Println("Error converting epoch year to int", err)
	}
	year += 2000 // doesn't work for pre Y2K

	date, err := strconv.ParseFloat(s[20:32], 32)
	if err != nil {
		fmt.Println("Error converting epoch day to float32", err)
	}

	//fmt.Printf("Satellite: %v, at epoch: %v", satnum, parse_date(year, date))
	return satnum, parse_date(year, date)
}

func parse_date(year int, day float64) time.Time {
	start := time.Date(year, 1, 1, 0, 0, 0, 0, time.UTC)
	intDays := int(day)
	remainder := day - float64(intDays)
	remainder_seconds := int(remainder * 60 * 60 * 24)

	// add the days
	result := start.AddDate(0, 0, intDays-1)
	// add the fractional day
	result = result.Add(time.Duration(remainder_seconds) * time.Second)

	return result
}
