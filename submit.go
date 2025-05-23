package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"net/http"
	"os"
	"skyscan/internal/database"
	"time"
)

func (c *apiConfig) handlerRecieveEntry(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Recieving thing")

	//                    'RA1':float(ra_first),
	//                    'DEC1':float(dec_first),
	//                    'RA2':float(ra_second),
	//                    'DEC2':float(dec_second),
	//                    'EXPTIME':float(exptime),
	//                    'EXPSTART':expstart,
	//                    'x_pix':cx, 'y_pix':cy}
	//"2025-04-17T04:38:51.946
	layout := "2006-01-02T15:04:05"

	type properties struct {
		RA1        float64     `json:"RA1"`
		DEC1       float64     `json:"DEC1"`
		RA2        float64     `json:"RA2"`
		DEC2       float64     `json:"DEC2"`
		EXPTIME    float64     `json:"EXPTIME"`
		EXPSTART   string      `json:"EXPSTART"`
		X_pix      float64     `json:"x_pix"`
		Y_pix      float64     `json:"y_pix"`
		IMG_width  int         `json:"width"`
		IMG_height int         `json:"height"`
		SATNUM     int         `json:"satnum"`
		IMG_data   [][]float32 `json:"image"`
	}

	decoder := json.NewDecoder(r.Body)
	fmt.Println(r.Body)
	myProperties := properties{}
	err := decoder.Decode(&myProperties)
	if err != nil {
		jsonRespondError(w, 500, "Something went wrong recieving json")
		fmt.Println(err)
		return
	}

	// Parse the string into a time.Time
	parsedTime, err := time.Parse(layout, myProperties.EXPSTART)
	if err != nil {
		fmt.Println("Error parsing time:", err)
		return
	}

	img := image.NewGray16(image.Rect(0, 0, 32, 32))
	//	for y := 0; y < imgData.Height; y++ {
	//		for x := 0; x < imgData.Width; x++ {
	//			img.Set(x, y, color.Gray{uint8(imgData.Image[y][x])})
	//		}
	//	}

	for i := range 32 {
		for j := range 32 {
			pixel_value := myProperties.IMG_data[i][j]
			img.Set(i, j, color.Gray16{uint16(255 * 255 * pixel_value)})
		}
	}

	buffer := new(bytes.Buffer)
	err = png.Encode(buffer, img)
	if err != nil {
		fmt.Println("Error converting img to byte array")
		return
	}

	base64Image := base64.StdEncoding.EncodeToString(buffer.Bytes())
	fmt.Println(myProperties)
	fmt.Println(myProperties.SATNUM)

	satnum := sql.NullInt32{Int32: 0, Valid: false}
	if myProperties.SATNUM > 0 {
		satnum.Int32 = int32(myProperties.SATNUM)
		satnum.Valid = true
	}

	// do the
	create_transient_params := database.CreateTransientParams{
		Expstart: parsedTime,
		Exptime:  myProperties.EXPTIME,
		Ra1:      myProperties.RA1,
		Ra2:      myProperties.RA2,
		Dec1:     myProperties.DEC1,
		Dec2:     myProperties.DEC2,
		Satnum:   satnum,
		Imgdata:  base64Image}

	fmt.Printf("Recieved (RA,DEC) = (%v,%v)->(%v,%v) at %v\n",
		myProperties.RA1, myProperties.DEC1,
		myProperties.RA2, myProperties.DEC2,
		parsedTime)
	fmt.Printf("Image data w:%v, h:%v, \n data: %v\n",
		myProperties.IMG_width,
		myProperties.IMG_height,
		base64Image)

	//	decode_image(base64Image)

	transient, err := c.dbQueries.CreateTransient(context.Background(), create_transient_params)
	if err != nil {
		log.Println("Error adding transient to database")
		return
	}
	fmt.Println("Created transient", transient)

	jsonRespondPayload(w, 200, "ok")

}

func convert_img_to_bytes(image [][]float32, width, height int) ([]byte, error) {

	//	buf := bytes.NewBuffer()
	buf := new(bytes.Buffer)

	for _, row := range image {
		for _, val := range row {
			//			fmt.Println(val)
			binary.Write(buf, binary.LittleEndian, val)
		}
	}

	// buf := make([]byte, 4*width*height, 4*width*height)
	//
	//	for i, row := range image {
	//		for j, val := range row {
	//			index := 4 * (i*width + j)
	//			buf[index] = byte(val)
	//		}
	//	}
	//
	return buf.Bytes(), nil
}

func decode_image(encoded_image string) {

	byte_image, err := base64.StdEncoding.DecodeString(encoded_image)
	if err != nil {
		fmt.Println("Error decoding base64 image data")
		return
	}

	img := image.NewGray16(image.Rect(0, 0, 32, 32))
	//	for y := 0; y < imgData.Height; y++ {
	//		for x := 0; x < imgData.Width; x++ {
	//			img.Set(x, y, color.Gray{uint8(imgData.Image[y][x])})
	//		}
	//	}

	reader := bytes.NewReader(byte_image)
	for i := range 32 {
		for j := range 32 {
			value := make([]byte, 4)
			reader.Read(value)
			bits := binary.LittleEndian.Uint32(value)
			float_value := math.Float32frombits(bits)
			//fmt.Println(float_value)
			img.Set(i, j, color.Gray16{uint16(255 * float_value)})
		}
	}

	err = saveImageToPNG(img, "test.png")
	if err != nil {
		fmt.Println("Saving image error", err)
	}

}

func saveImageToPNG(img image.Image, filename string) error {
	// Create or truncate the file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Encode the image to PNG and write to file
	return png.Encode(file, img)
}
