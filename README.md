Examples for PEZY-SCx processors (For PZSDK v4.1)
=================================================

How to build programs
---------------------

You can build program to change current derectory to project directory.

To change directory, use the `cd` command:

```
$ cd 0_Simple/pzcAdd
```

To build program, use the `make` command:

```
$ make
```

How to execute programs
-----------------------

To execute program, just run `make run` command

```
$ make run
```

List of Samples
===============

This sample contains following examples.

Environments of Makefile
========================

| Variables        | Descriptions                                                                                            |
|------------------|---------------------------------------------------------------------------------------------------------|
| PZC\_TARGET\_ARCH| To change target PEZY architectures. For PEZY-SC use `sc1-64`. For PEZY-SC2 use `sc2`. default is `sc2` |
