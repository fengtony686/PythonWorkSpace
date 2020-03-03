# By SharzyL
package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
)

const server string = "localhost:9999"

func communicate(conn net.Conn, sentMsg string) {
	conn.Write([]byte(sentMsg))
	buf := make([]byte, 200)
	n, err := conn.Read(buf)
	if err != nil {
		log.Println("Server closed the connection")
		os.Exit(1)
	}
	fmt.Print(">>>receive msg: \n    ", 
			  string(buf[0:n]),
			   "\n")
}

func stripString(str string) string {
	if strings.HasSuffix(str, "\r\n") {
		l := len(str)
		return str[0:l - 2]
	}
	return str
}

func main() {
	conn, err := net.Dial("tcp", server)
	if err != nil {
		log.Println("dial error:", err)
		os.Exit(1)
	}
	defer conn.Close()
	log.Println("dial ok")

	inputReader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(">>> send msg: \n    ")
		msg, err := inputReader.ReadString('\n')
		if err != nil {
			log.Println("exceptional input", msg)
			continue
		}
		msg = stripString(msg)
		communicate(conn, msg)
	}
}
