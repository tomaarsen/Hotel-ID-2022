settings {
   logfile    = "/tmp/lsyncd.log",
   statusFile = "/tmp/lsyncd.status",
   nodaemon   = true
}

sync {
   default.rsyncssh,
   source       ="/path/to/tiny-voxceleb-skeleton/repo/on/your/local/computer",
   host         ="your_science_ru_username@cn99.science.ru.nl",
   excludeFrom  =".gitignore",
   targetdir    ="/home/your_science_ru_username/remote/repo/tiny-voxceleb-skeleton",
   delay        = 0,
   rsync = {
     archive    = true,
     compress   = false,
     whole_file = false
   }
}