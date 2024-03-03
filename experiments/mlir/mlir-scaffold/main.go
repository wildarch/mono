package main

import (
	"embed"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"text/template"
)

type DialectConfig struct {
	BaseDir            string
	DialectName        string
	DialectNameScream  string
	Namespace          string
	DialectLibName     string
	ProjectDescription string
	OverwriteExisting  bool
}

var baseDir = flag.String("base-dir", "/tmp/mlir-scaffold-test", "Base directory for the new dialect to be created under")
var dialectName = flag.String("name", "Test", "Name for the new dialect, in CamelCase")
var projectDescription = flag.String("project-description", "TODO: write a description", "CMake project description")
var overwrite = flag.Bool("overwrite", false, "Overwrite existing files")

//go:embed templates templates/.gitignore.template
var templates embed.FS

func createFromTemplateIfNotExists(config *DialectConfig, path string) error {
	outputPath := fmt.Sprintf("%s/%s", config.BaseDir, path)
	// Update names where needed
	outputPath = strings.ReplaceAll(outputPath, "Scaffold", config.DialectName)

	flags := os.O_RDWR | os.O_CREATE
	if config.OverwriteExisting {
		flags |= os.O_TRUNC
	} else {
		flags |= os.O_EXCL
	}
	outFile, err := os.OpenFile(outputPath, flags, 0666)
	if os.IsExist(err) {
		log.Printf("skip %s because it already exists", outputPath)
		return nil
	} else if err != nil {
		return fmt.Errorf("error opening output file %s: %w", outputPath, err)
	}
	defer outFile.Close()

	templatePath := fmt.Sprintf("templates/%s.template", path)
	tmpl, err := template.ParseFS(templates, templatePath)
	if err != nil {
		return fmt.Errorf("error parsing template: %w", err)
	}

	err = tmpl.Execute(outFile, config)
	if err != nil {
		return fmt.Errorf("error writing template to %s: %w", outputPath, err)
	}

	log.Printf("create %s", outputPath)
	return nil
}

func main() {
	flag.Parse()

	config := DialectConfig{
		BaseDir:            *baseDir,
		DialectName:        *dialectName,
		DialectNameScream:  strings.ToUpper(*dialectName),
		Namespace:          strings.ToLower(*dialectName),
		DialectLibName:     strings.ToLower(*dialectName),
		ProjectDescription: *projectDescription,
		OverwriteExisting:  *overwrite,
	}

	files := []string{
		".gitignore",
		"CMakeLists.txt",
		"configure.sh",
		"include/CMakeLists.txt",
		"include/ScaffoldDialect.h",
		"include/ScaffoldDialect.td",
		"include/ScaffoldOps.h",
		"include/ScaffoldOps.td",
		"include/ScaffoldTypes.h",
		"include/ScaffoldTypes.td",
		"lib/CMakeLists.txt",
		"lib/ScaffoldDialect.cpp",
		"lib/ScaffoldOps.cpp",
		"lib/ScaffoldTypes.cpp",
		"opt/CMakeLists.txt",
		"opt/opt.cpp",
	}
	os.MkdirAll(config.BaseDir+"/include", os.ModePerm)
	os.MkdirAll(config.BaseDir+"/lib", os.ModePerm)
	os.MkdirAll(config.BaseDir+"/opt", os.ModePerm)

	for _, file := range files {
		err := createFromTemplateIfNotExists(&config, file)
		if err != nil {
			log.Fatalf("failed to create project file %s: %s", file, err)
		}
	}
}
