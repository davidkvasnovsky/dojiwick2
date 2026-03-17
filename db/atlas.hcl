env "local" {
  url = getenv("DOJIWICK_DB_URL")
  dev = getenv("DOJIWICK_DB_URL")
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
  dev = getenv("DOJIWICK_TEST_DB_URL")
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
