# dev database major must match the postgres image in compose.yaml
env "local" {
  url = getenv("DOJIWICK_DB_URL")
  dev = "docker://postgres/18/dev"
  src = "file://db/schema.sql"

  migration {
    dir = "file://db/migrations"
  }

  format {
    migrate {
      diff = "{{ sql . \"  \" }}"
    }
  }
}

env "test" {
  url = getenv("DOJIWICK_TEST_DB_URL")
  dev = "docker://postgres/18/dev"
  src = "file://db/schema.sql"

  migration {
    dir = "file://db/migrations"
  }

  format {
    migrate {
      diff = "{{ sql . \"  \" }}"
    }
  }
}
