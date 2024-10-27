package main

import (
	"context"
	_ "embed"
	"html/template"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"github.com/oracle/oci-go-sdk/v65/common"
	"github.com/oracle/oci-go-sdk/v65/objectstorage"
)

const (
	templatesDir = "templates"
	bucketName   = "medialib"
)

var (
	//go:embed templates/list.html
	listHtml string

	//go:embed templates/play.html
	playHtml string
)

func getNamespace(ctx context.Context, c objectstorage.ObjectStorageClient) string {
	request := objectstorage.GetNamespaceRequest{}
	r, err := c.GetNamespace(ctx, request)
	if err != nil {
		log.Fatalf("failed to get namespace: %v", err)
	}

	return *r.Value
}

type listHandler struct {
	osClient   *objectstorage.ObjectStorageClient
	namespace  string
	bucketName string
	tmpl       *template.Template
}

func (lh listHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	request := objectstorage.ListObjectsRequest{
		NamespaceName: &lh.namespace,
		BucketName:    &lh.bucketName,
	}

	objectsResponse, err := lh.osClient.ListObjects(r.Context(), request)
	if err != nil {
		log.Fatalf("failed to list objects: %v", err)
	}

	err = lh.tmpl.Execute(w, objectsResponse)
	if err != nil {
		log.Printf("failed to render list template: %v", err)
	}
}

type playHandler struct {
	osClient   *objectstorage.ObjectStorageClient
	namespace  string
	bucketName string
	tmpl       *template.Template
}

type playData struct {
	Name     string
	VideoURL string
	SubURL   string
}

func (ph playHandler) makePublicURL(ctx context.Context, file string) (string, error) {
	requestName := "medialib-play"
	expires := common.SDKTime{Time: time.Now().Add(time.Hour * 2)}
	par := objectstorage.CreatePreauthenticatedRequestRequest{
		NamespaceName: &ph.namespace,
		BucketName:    &ph.bucketName,
		CreatePreauthenticatedRequestDetails: objectstorage.CreatePreauthenticatedRequestDetails{
			Name:        &requestName,
			ObjectName:  &file,
			AccessType:  objectstorage.CreatePreauthenticatedRequestDetailsAccessTypeObjectread,
			TimeExpires: &expires,
		},
	}

	res, err := ph.osClient.CreatePreauthenticatedRequest(ctx, par)
	if err != nil {
		return "", err
	}

	return ph.osClient.Endpoint() + *res.AccessUri, nil
}

func (ph playHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	file := r.URL.Query().Get("file")

	ext := filepath.Ext(file)
	name := file[0 : len(file)-len(ext)]

	video, err := ph.makePublicURL(r.Context(), file)
	if err != nil {
		log.Fatalf("failed to create url for video: %v", err)
	}

	subFile := name + ".en.vtt"
	sub, err := ph.makePublicURL(r.Context(), subFile)
	if err != nil {
		log.Fatalf("failed to create url for subtitles: %v", err)
	}

	data := playData{
		Name:     name,
		VideoURL: video,
		SubURL:   sub,
	}

	err = ph.tmpl.Execute(w, data)
	if err != nil {
		log.Fatalf("failed to render play template: %v", err)
	}
}

func main() {
	// Setup the object storage client
	configProvider := common.DefaultConfigProvider()
	osClient, err := objectstorage.NewObjectStorageClientWithConfigurationProvider(configProvider)
	if err != nil {
		log.Fatalf("failed to create OCI client: %v", err)
	}

	// Configure region
	osClient.SetRegion(string(common.RegionEUAmsterdam1))

	// Namespace
	ctx := context.Background()
	namespace := getNamespace(ctx, osClient)

	// HTTP server
	mux := http.NewServeMux()

	listTemplate := template.Must(template.New("list").Parse(listHtml))
	lh := listHandler{osClient: &osClient, namespace: namespace, bucketName: bucketName, tmpl: listTemplate}
	mux.Handle("/", lh)

	playTemplate := template.Must(template.New("play").Parse(playHtml))
	ph := playHandler{osClient: &osClient, namespace: namespace, bucketName: bucketName, tmpl: playTemplate}
	mux.Handle("/play", ph)

	http.ListenAndServe(":8001", mux)
}
