// build.sbt

name := "bigdl_regression_example"

version := "0.1.0"

scalaVersion := "2.11.8"

scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Xlint")

unmanagedJars in Compile += file("lib/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar")

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "2.0.2",
    "org.apache.spark" %% "spark-mllib" % "2.0.2"
)
