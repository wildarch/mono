package main

import (
	"bufio"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path"

	_ "modernc.org/sqlite"
)

type Repo struct {
	db *sql.DB
}

func (r *Repo) Initialize() error {
	_, err := r.db.Exec(`
		PRAGMA synchronous = OFF;
		PRAGMA journal_mode = MEMORY;
		DROP TABLE IF EXISTS Source;
		CREATE TABLE Source(sha256 TEXT PRIMARY KEY, data BLOB NOT NULL);
		DROP TABLE IF EXISTS File;
		CREATE TABLE File(path TEXT PRIMARY KEY, sha256 TEXT, ruleId);
	`)
	if err != nil {
		return err
	}

	return nil
}

func (r *Repo) AddSource(p string, d []byte) error {
	h := sha256.New()
	h.Write(d)
	sid := fmt.Sprintf("%x", h.Sum(nil))
	_, err := r.db.Exec("INSERT INTO Source VALUES (?, ?)", sid, d)
	if err != nil {
		return err
	}

	_, err = r.db.Exec("INSERT INTO File VALUES (?, ?, NULL)", p, sid)
	if err != nil {
		return err
	}

	return nil
}

func (r *Repo) getSource(sha256 string) ([]byte, error) {
	rows, err := r.db.Query("SELECT data FROM Source WHERE sha256 = ?", sha256)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var data []byte
		if err := rows.Scan(&data); err != nil {
			return nil, err
		}

		return data, nil
	}

	return nil, fmt.Errorf("no source file for sha256 %s", sha256)
}

func (r *Repo) Build(root string, p string) error {
	rows, err := r.db.Query("SELECT sha256, ruleId FROM File WHERE path = ?", p)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var sha256 sql.NullString
		var ruleId sql.NullInt64
		if err := rows.Scan(&sha256, &ruleId); err != nil {
			return err
		}

		if ruleId.Valid {
			return fmt.Errorf("TODO: build")
		} else {
			// Must be a source file
			data, err := r.getSource(sha256.String)
			if err != nil {
				return err
			}

			out := path.Join(root, p)

			// Ensure target directory exists
			dir, _ := path.Split(out)
			if err := os.MkdirAll(dir, 0755); err != nil {
				return err
			}

			if err := os.WriteFile(out, data, 0755); err != nil {
				return err
			}

			log.Printf("wrote source file %s", out)
			return nil
		}
	}

	return fmt.Errorf("no such target: %s", p)
}

func addGitRepo(root string, repo *Repo) error {
	// Get list of files from git
	git := exec.Command("git", "ls-files")
	git.Dir = root
	out, err := git.StdoutPipe()
	if err != nil {
		return err
	}

	if err := git.Start(); err != nil {
		return err
	}

	scanner := bufio.NewScanner(out)
	for scanner.Scan() {
		file := scanner.Text()
		f, err := os.Open(file)
		if err != nil {
			return err
		}

		defer f.Close()
		data, err := io.ReadAll(f)
		if err != nil {
			return err
		}

		repo.AddSource(file, data)
	}

	if err := git.Wait(); err != nil {
		return err
	}

	return nil
}

func main() {
	db, err := sql.Open("sqlite", "/tmp/dabu.sqlite3")
	if err != nil {
		log.Fatal(err)
	}

	repo := &Repo{db}
	if err := repo.Initialize(); err != nil {
		log.Fatal(err)
	}

	if err := addGitRepo(".", repo); err != nil {
		log.Fatal(err)
	}

	// Create root
	root := "/tmp/dabu-root"
	if err := os.RemoveAll(root); err != nil {
		log.Fatal(err)
	}

	if err := os.Mkdir(root, 0755); err != nil {
		log.Fatal(err)
	}

	sourceDir := "experiments/dabu/cmake-hello"
	cmakeLists := path.Join(sourceDir, "CMakeLists.txt")
	if err := repo.Build(root, cmakeLists); err != nil {
		log.Fatal(err)
	}

	if err := db.Close(); err != nil {
		log.Fatal(err)
	}
}
