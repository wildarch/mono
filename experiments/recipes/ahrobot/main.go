package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/chromedp/chromedp"
)

func main() {
	homedir, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("Cannot get user home directory: %w", err)
	}
	datadir := fmt.Sprintf("%s/.config/ahrobot/chrome_userdata", homedir)
	// create context
	opts := []chromedp.ExecAllocatorOption{
		chromedp.NoFirstRun,
		chromedp.NoDefaultBrowserCheck,
		chromedp.UserDataDir(datadir),
	}
	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()
	ctx, cancel := chromedp.NewContext(
		allocCtx,
		// chromedp.WithDebugf(log.Printf),
	)
	defer cancel()

	pb := "wi461685"

	if err := AddProduct(ctx, pb); err != nil {
		log.Fatal(err)
	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Press ENTER to terminate")
	reader.ReadString('\n')
}

func AddProduct(ctx context.Context, code string) error {
	url := fmt.Sprintf("https://www.ah.nl/producten/product/%s", code)

	if err := chromedp.Run(ctx, chromedp.Navigate(url)); err != nil {
		return err
	}

	if err := chromedp.Run(ctx, chromedp.Click("button[data-testhook=\"product-plus\"]")); err != nil {
		return err
	}

	return nil
}
