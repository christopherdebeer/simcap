
find . -type f -print0 | while IFS= read -r -d $'\0' file;
  do echo "$file" ;
  cat $file | jq -r '(map(keys) | add | unique) as $cols | map(. as $row | $cols | map($row[.])) as $rows | $cols, $rows[] | @csv' > $file.csv;
done
