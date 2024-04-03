# 安装依赖


```python
!pip install jieba>=0.42.1
!pip install ruamel_yaml>=0.18.6
!pip install rouge_chinese>=1.0.3
!pip install jupyter>=1.0.0
!pip install datasets>=2.17.1
!pip install peft>=0.10.0
!pip install transformers>=4.38.1
!pip install deepspeed==0.13.1
!pip install mpi4py>=3.1.5
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mLooking in indexes: http://mirrors.aliyun.com/pypi/simple
    Collecting deepspeed==0.13.1
      Downloading http://mirrors.aliyun.com/pypi/packages/ce/f8/3b63074f1841dd6cf5414decf9c4a338a34915d8491bb719608ec56cc3bb/deepspeed-0.13.1.tar.gz (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting hjson
      Downloading http://mirrors.aliyun.com/pypi/packages/1f/7f/13cd798d180af4bf4c0ceddeefba2b864a63c71645abc0308b768d67bb81/hjson-3.1.0-py3-none-any.whl (54 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.0/54.0 kB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting ninja
      Downloading http://mirrors.aliyun.com/pypi/packages/6d/92/8d7aebd4430ab5ff65df2bfee6d5745f95c004284db2d8ca76dcbfd9de47/ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m307.2/307.2 kB[0m [31m2.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: numpy in ./miniconda3/lib/python3.10/site-packages (from deepspeed==0.13.1) (1.26.3)
    Requirement already satisfied: packaging>=20.0 in ./miniconda3/lib/python3.10/site-packages (from deepspeed==0.13.1) (23.2)
    Requirement already satisfied: psutil in ./miniconda3/lib/python3.10/site-packages (from deepspeed==0.13.1) (5.9.7)
    Collecting py-cpuinfo
      Downloading http://mirrors.aliyun.com/pypi/packages/e0/a9/023730ba63db1e494a271cb018dcd361bd2c917ba7004c3e49d5daf795a2/py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
    Collecting pydantic
      Downloading http://mirrors.aliyun.com/pypi/packages/e5/f3/8296f550276194a58c5500d55b19a27ae0a5a3a51ffef66710c58544b32d/pydantic-2.6.4-py3-none-any.whl (394 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m394.9/394.9 kB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting pynvml
      Downloading http://mirrors.aliyun.com/pypi/packages/5b/9c/adb8070059caaa15d5a572b66bccd95900d8c1b9fa54d6ecea6ae97448d1/pynvml-11.5.0-py3-none-any.whl (53 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.1/53.1 kB[0m [31m351.9 kB/s[0m eta [36m0:00:00[0m [36m0:00:01[0m
    [?25hRequirement already satisfied: torch in ./miniconda3/lib/python3.10/site-packages (from deepspeed==0.13.1) (2.1.2+cu121)
    Requirement already satisfied: tqdm in ./miniconda3/lib/python3.10/site-packages (from deepspeed==0.13.1) (4.64.1)
    Collecting annotated-types>=0.4.0
      Downloading http://mirrors.aliyun.com/pypi/packages/28/78/d31230046e58c207284c6b2c4e8d96e6d3cb4e52354721b944d3e1ee4aa5/annotated_types-0.6.0-py3-none-any.whl (12 kB)
    Collecting pydantic-core==2.16.3
      Downloading http://mirrors.aliyun.com/pypi/packages/b8/be/a3c2edde00afcf5cdc0fb710ce0289f5af776273f420b4486cf005c94b57/pydantic_core-2.16.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: typing-extensions>=4.6.1 in ./miniconda3/lib/python3.10/site-packages (from pydantic->deepspeed==0.13.1) (4.9.0)
    Requirement already satisfied: filelock in ./miniconda3/lib/python3.10/site-packages (from torch->deepspeed==0.13.1) (3.13.1)
    Requirement already satisfied: jinja2 in ./miniconda3/lib/python3.10/site-packages (from torch->deepspeed==0.13.1) (3.1.2)
    Requirement already satisfied: networkx in ./miniconda3/lib/python3.10/site-packages (from torch->deepspeed==0.13.1) (3.2.1)
    Requirement already satisfied: fsspec in ./miniconda3/lib/python3.10/site-packages (from torch->deepspeed==0.13.1) (2023.12.2)
    Requirement already satisfied: sympy in ./miniconda3/lib/python3.10/site-packages (from torch->deepspeed==0.13.1) (1.12)
    Requirement already satisfied: triton==2.1.0 in ./miniconda3/lib/python3.10/site-packages (from torch->deepspeed==0.13.1) (2.1.0)
    Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/lib/python3.10/site-packages (from jinja2->torch->deepspeed==0.13.1) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in ./miniconda3/lib/python3.10/site-packages (from sympy->torch->deepspeed==0.13.1) (1.3.0)
    Building wheels for collected packages: deepspeed
      Building wheel for deepspeed (setup.py) ... [?25ldone
    [?25h  Created wheel for deepspeed: filename=deepspeed-0.13.1-py3-none-any.whl size=1350309 sha256=70e0959c492b665ca88ebd84c206489394f7911dafe8f2ce28bc20ef98ffd27d
      Stored in directory: /root/.cache/pip/wheels/ed/e9/2d/c8a60619ddb3cafe92b83268059a6ba61cf3050b5aaf10c1c7
    Successfully built deepspeed
    Installing collected packages: py-cpuinfo, ninja, hjson, pynvml, pydantic-core, annotated-types, pydantic, deepspeed
    Successfully installed annotated-types-0.6.0 deepspeed-0.13.1 hjson-3.1.0 ninja-1.11.1.1 py-cpuinfo-9.0.0 pydantic-2.6.4 pydantic-core-2.16.3 pynvml-11.5.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m  [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mBuilding wheel for mpi4py [0m[1;32m([0m[32mpyproject.toml[0m[1;32m)[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[148 lines of output][0m
      [31m   [0m running bdist_wheel
      [31m   [0m running build
      [31m   [0m running build_src
      [31m   [0m running build_py
      [31m   [0m creating build
      [31m   [0m creating build/lib.linux-x86_64-cpython-310
      [31m   [0m creating build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/__init__.py -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/__main__.py -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/bench.py -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/run.py -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m creating build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/__init__.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/__main__.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/_base.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/_core.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/_lib.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/aplus.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/pool.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/server.py -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m creating build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m copying src/mpi4py/util/__init__.py -> build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m copying src/mpi4py/util/dtlib.py -> build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m copying src/mpi4py/util/pkl5.py -> build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m copying src/mpi4py/MPI.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/__init__.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/__main__.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/bench.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/dl.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/run.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/py.typed -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/MPI.pxd -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/__init__.pxd -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m copying src/mpi4py/libmpi.pxd -> build/lib.linux-x86_64-cpython-310/mpi4py
      [31m   [0m creating build/lib.linux-x86_64-cpython-310/mpi4py/include
      [31m   [0m creating build/lib.linux-x86_64-cpython-310/mpi4py/include/mpi4py
      [31m   [0m copying src/mpi4py/include/mpi4py/mpi4py.MPI.h -> build/lib.linux-x86_64-cpython-310/mpi4py/include/mpi4py
      [31m   [0m copying src/mpi4py/include/mpi4py/mpi4py.MPI_api.h -> build/lib.linux-x86_64-cpython-310/mpi4py/include/mpi4py
      [31m   [0m copying src/mpi4py/include/mpi4py/mpi4py.h -> build/lib.linux-x86_64-cpython-310/mpi4py/include/mpi4py
      [31m   [0m copying src/mpi4py/include/mpi4py/mpi4py.i -> build/lib.linux-x86_64-cpython-310/mpi4py/include/mpi4py
      [31m   [0m copying src/mpi4py/include/mpi4py/mpi.pxi -> build/lib.linux-x86_64-cpython-310/mpi4py/include/mpi4py
      [31m   [0m copying src/mpi4py/futures/__init__.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/__main__.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/_core.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/_lib.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/aplus.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/pool.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/futures/server.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/futures
      [31m   [0m copying src/mpi4py/util/__init__.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m copying src/mpi4py/util/dtlib.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m copying src/mpi4py/util/pkl5.pyi -> build/lib.linux-x86_64-cpython-310/mpi4py/util
      [31m   [0m running build_clib
      [31m   [0m MPI configuration: [mpi] from 'mpi.cfg'
      [31m   [0m checking for library 'lmpe' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -llmpe -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -llmpe: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m building 'mpe' dylib library
      [31m   [0m creating build/temp.linux-x86_64-cpython-310
      [31m   [0m creating build/temp.linux-x86_64-cpython-310/src
      [31m   [0m creating build/temp.linux-x86_64-cpython-310/src/lib-pmpi
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c src/lib-pmpi/mpe.c -o build/temp.linux-x86_64-cpython-310/src/lib-pmpi/mpe.o
      [31m   [0m creating build/lib.linux-x86_64-cpython-310/mpi4py/lib-pmpi
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -shared -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,--no-as-needed build/temp.linux-x86_64-cpython-310/src/lib-pmpi/mpe.o -o build/lib.linux-x86_64-cpython-310/mpi4py/lib-pmpi/libmpe.so
      [31m   [0m checking for library 'vt-mpi' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -lvt-mpi -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -lvt-mpi: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m checking for library 'vt.mpi' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -lvt.mpi -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -lvt.mpi: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m building 'vt' dylib library
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c src/lib-pmpi/vt.c -o build/temp.linux-x86_64-cpython-310/src/lib-pmpi/vt.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -shared -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,--no-as-needed build/temp.linux-x86_64-cpython-310/src/lib-pmpi/vt.o -o build/lib.linux-x86_64-cpython-310/mpi4py/lib-pmpi/libvt.so
      [31m   [0m checking for library 'vt-mpi' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -lvt-mpi -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -lvt-mpi: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m checking for library 'vt.mpi' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -lvt.mpi -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -lvt.mpi: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m building 'vt-mpi' dylib library
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c src/lib-pmpi/vt-mpi.c -o build/temp.linux-x86_64-cpython-310/src/lib-pmpi/vt-mpi.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -shared -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,--no-as-needed build/temp.linux-x86_64-cpython-310/src/lib-pmpi/vt-mpi.o -o build/lib.linux-x86_64-cpython-310/mpi4py/lib-pmpi/libvt-mpi.so
      [31m   [0m checking for library 'vt-hyb' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -lvt-hyb -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -lvt-hyb: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m checking for library 'vt.ompi' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -lvt.ompi -o _configtest
      [31m   [0m /root/miniconda3/compiler_compat/ld: cannot find -lvt.ompi: No such file or directory
      [31m   [0m collect2: error: ld returned 1 exit status
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m building 'vt-hyb' dylib library
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -c src/lib-pmpi/vt-hyb.c -o build/temp.linux-x86_64-cpython-310/src/lib-pmpi/vt-hyb.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -shared -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,--no-as-needed build/temp.linux-x86_64-cpython-310/src/lib-pmpi/vt-hyb.o -o build/lib.linux-x86_64-cpython-310/mpi4py/lib-pmpi/libvt-hyb.so
      [31m   [0m running build_ext
      [31m   [0m MPI configuration: [mpi] from 'mpi.cfg'
      [31m   [0m checking for dlopen() availability ...
      [31m   [0m checking for header 'dlfcn.h' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -I/root/miniconda3/include/python3.10 -c _configtest.c -o _configtest.o
      [31m   [0m success!
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m success!
      [31m   [0m checking for library 'dl' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -I/root/miniconda3/include/python3.10 -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -Lbuild/temp.linux-x86_64-cpython-310 -ldl -o _configtest
      [31m   [0m success!
      [31m   [0m removing: _configtest.c _configtest.o _configtest
      [31m   [0m checking for function 'dlopen' ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -I/root/miniconda3/include/python3.10 -c _configtest.c -o _configtest.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat _configtest.o -Lbuild/temp.linux-x86_64-cpython-310 -ldl -o _configtest
      [31m   [0m success!
      [31m   [0m removing: _configtest.c _configtest.o _configtest
      [31m   [0m building 'mpi4py.dl' extension
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -DHAVE_DLFCN_H=1 -DHAVE_DLOPEN=1 -I/root/miniconda3/include/python3.10 -c src/dynload.c -o build/temp.linux-x86_64-cpython-310/src/dynload.o
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -shared -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib -Wl,-rpath,/root/miniconda3/lib -Wl,-rpath-link,/root/miniconda3/lib -L/root/miniconda3/lib build/temp.linux-x86_64-cpython-310/src/dynload.o -Lbuild/temp.linux-x86_64-cpython-310 -ldl -o build/lib.linux-x86_64-cpython-310/mpi4py/dl.cpython-310-x86_64-linux-gnu.so
      [31m   [0m checking for MPI compile and link ...
      [31m   [0m gcc -pthread -B /root/miniconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/miniconda3/include -fPIC -O2 -isystem /root/miniconda3/include -fPIC -I/root/miniconda3/include/python3.10 -c _configtest.c -o _configtest.o
      [31m   [0m _configtest.c:2:10: fatal error: mpi.h: No such file or directory
      [31m   [0m     2 | #include <mpi.h>
      [31m   [0m       |          ^~~~~~~
      [31m   [0m compilation terminated.
      [31m   [0m failure.
      [31m   [0m removing: _configtest.c _configtest.o
      [31m   [0m error: Cannot compile MPI programs. Check your configuration!!!
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [31m  ERROR: Failed building wheel for mpi4py[0m[31m
    [0m[31mERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects[0m[31m
    [0m


```python
pip install typer
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Collecting typer
      Downloading http://mirrors.aliyun.com/pypi/packages/d8/21/6a9c5c3363e30345346f59d236fd416730c2ac7802abe3609fed89d46ac1/typer-0.12.0-py3-none-any.whl (5.6 kB)
    Collecting typer-cli==0.12.0
      Downloading http://mirrors.aliyun.com/pypi/packages/c1/11/a0a5c4ff576299b57a5012a618291ea3f64f75e187c6dec276944d5820a5/typer_cli-0.12.0-py3-none-any.whl (3.0 kB)
    Collecting typer-slim[standard]==0.12.0
      Downloading http://mirrors.aliyun.com/pypi/packages/9e/8d/cd24db348ffdec4e2331c293f63eb8492956879c0925dd05d1e747b49cc7/typer_slim-0.12.0-py3-none-any.whl (46 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.8/46.8 kB[0m [31m291.1 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: typing-extensions>=3.7.4.3 in ./miniconda3/lib/python3.10/site-packages (from typer-slim[standard]==0.12.0->typer) (4.9.0)
    Collecting click>=8.0.0
      Downloading http://mirrors.aliyun.com/pypi/packages/00/2e/d53fa4befbf2cfa713304affc7ca780ce4fc1fd8710527771b58311a3229/click-8.1.7-py3-none-any.whl (97 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m97.9/97.9 kB[0m [31m296.5 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting shellingham>=1.3.0
      Downloading http://mirrors.aliyun.com/pypi/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
    Collecting rich>=10.11.0
      Downloading http://mirrors.aliyun.com/pypi/packages/87/67/a37f6214d0e9fe57f6ae54b2956d550ca8365857f42a1ce0392bb21d9410/rich-13.7.1-py3-none-any.whl (240 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m240.7/240.7 kB[0m [31m290.3 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: pygments<3.0.0,>=2.13.0 in ./miniconda3/lib/python3.10/site-packages (from rich>=10.11.0->typer-slim[standard]==0.12.0->typer) (2.17.2)
    Collecting markdown-it-py>=2.2.0
      Downloading http://mirrors.aliyun.com/pypi/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m87.5/87.5 kB[0m [31m286.3 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting mdurl~=0.1
      Downloading http://mirrors.aliyun.com/pypi/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Installing collected packages: shellingham, mdurl, click, typer-slim, markdown-it-py, rich, typer-cli, typer
    Successfully installed click-8.1.7 markdown-it-py-3.0.0 mdurl-0.1.2 rich-13.7.1 shellingham-1.5.4 typer-0.12.0 typer-cli-0.12.0 typer-slim-0.12.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
pip install nltk
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Collecting nltk
      Downloading http://mirrors.aliyun.com/pypi/packages/a6/0a/0d20d2c0f16be91b9fa32a77b76c60f9baf6eba419e5ef5deca17af9c582/nltk-3.8.1-py3-none-any.whl (1.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m271.5 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: tqdm in ./miniconda3/lib/python3.10/site-packages (from nltk) (4.64.1)
    Collecting joblib
      Downloading http://mirrors.aliyun.com/pypi/packages/10/40/d551139c85db202f1f384ba8bcf96aca2f329440a844f924c8a0040b6d02/joblib-1.3.2-py3-none-any.whl (302 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.2/302.2 kB[0m [31m273.5 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: click in ./miniconda3/lib/python3.10/site-packages (from nltk) (8.1.7)
    Requirement already satisfied: regex>=2021.8.3 in ./miniconda3/lib/python3.10/site-packages (from nltk) (2023.12.25)
    Installing collected packages: joblib, nltk
    Successfully installed joblib-1.3.2 nltk-3.8.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
pip install sentencepiece
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Requirement already satisfied: pip in ./miniconda3/lib/python3.10/site-packages (22.3.1)
    Collecting install
      Downloading http://mirrors.aliyun.com/pypi/packages/4d/c8/8cbca135f9e167810756ea2bc34b028501936675fcbd7dadccf752fa4622/install-1.3.5-py3-none-any.whl (3.2 kB)
    Collecting sentencepiece
      Downloading http://mirrors.aliyun.com/pypi/packages/a6/27/33019685023221ca8ed98e8ceb7ae5e166032686fa3662c68f1f1edf334e/sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m279.0 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hInstalling collected packages: sentencepiece, install
    Successfully installed install-1.3.5 sentencepiece-0.2.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.


# 准备训练数据集


```python
mkdir data
```

    mkdir: cannot create directory ‘data’: File exists



```python
import tarfile

# Specify the path to your tar.gz file and the output directory
tar_path = 'data/AdvertiseGen.tar.gz'
output_dir = './data/'

# Open the tar.gz file
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=output_dir)
```


```python
ll data/AdvertiseGen
```

    total 52992
    -rw-rw-r-- 1 1004   498394 Aug 16  2021 dev.json
    -rw-rw-r-- 1 1004 53763280 Aug 16  2021 train.json


# 查看样例数据


```python
import json
import random

# 加载JSON文件
file_path = 'data/AdvertiseGen/dev.json'  # JSON文件的路径

data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 检查数据长度，如果小于10，则取全部数据
sample_size = min(10, len(data))

# 随机选择10条或者全部数据（如果数据少于10条）
sample_data = random.sample(data, sample_size)

# 打印选中的记录
for item in sample_data:
    print(item)

```

    {'content': '类型#裙*颜色#白色*风格#性感*裙款式#露肩', 'summary': '女生衣柜里面必不可少的一条裙子就是白色的裙子，但是单调的纯白从来都不会是有时尚嗅觉的女孩的首选。这样一条蓝白混搭的小香风裙子十分完美的诠释了所有女孩子心里面白色裙子该有的样子。个性的旗袍领口看上去十分的别致出色，很有中国古典的优雅感觉。侧边的露肩设计看起来无比的性感优雅，也不会太过于暴露。'}
    {'content': '类型#裤*版型#宽松*裤长#九分裤*裤型#直筒裤', 'summary': '来自于<UNK>的这条宽松褶位九分裤，采用的是宽松直筒的版型设计，这种版型的包容性极佳，不仅不挑身材，还能起到一定的修饰不完美腿型效果，而前腰的打褶细节，则使得裤子更为立体。再添加可拆卸织带进行点缀，轻松就能凹出自主造型。'}
    {'content': '类型#裙*版型#宽松*版型#显瘦*颜色#黑色*裙腰型#高腰*裙袖型#喇叭袖', 'summary': '这款孕妇裙采用黑色的主色，黑色有视觉显瘦的效果。宽松的领口将脖颈修饰的更加修长。喇叭袖的设计可以遮挡手臂<UNK>的问题。高腰的版型是为了不让凸起的小肚子有紧绷难受的感觉。'}
    {'content': '类型#裙*版型#显瘦*版型#h*图案#波点*图案#印花*裙下摆#花边*裙长#连衣裙*裙领型#圆领', 'summary': '这款由itmichaa推出的连衣裙，修身h版型设计，穿搭上身有着显瘦的效果，适合各种身材穿搭。花边圆领的设计，个性又时髦，又能巧妙的修饰出小巧迷人的脸型。衣身通体以波点印花图案点缀，时尚而新颖，也为衣身带来了丰富的视觉看点。'}
    {'content': '类型#裙*裙下摆#花边*裙长#半身裙*裙款式#口袋*裙款式#抽褶*裙款式#对称', 'summary': '设计师用花边的设计，在半身裙的后面做了两个夸张的对称口袋，设计很特别。用褶皱花边最大程度上打造视觉上的甜美感觉，并没有显得很突兀。'}
    {'content': '类型#裙*材质#蕾丝*图案#条纹*图案#蕾丝*裙款式#勾花镂空*裙款式#收腰', 'summary': '这件裙子的颜色本身就够惹眼了，所以在鞋子、包包和其他配饰上不用太费心，简单些就好。竖条纹的设计让身材更加修长，肩部和裙摆的镂空蕾丝，给人优雅朦胧的感觉。收腰的设计不会凸显小腹，还能显出傲人的身材。'}
    {'content': '类型#裙*风格#淑女*风格#英伦*风格#复古*风格#文艺*图案#格子*图案#复古*裙下摆#荷叶边*裙长#连衣裙*裙领型#娃娃领*裙衣门襟#系带*裙款式#抽绳', 'summary': '最近时尚界大玩复古风潮，90年代的复古裙型，又再度流行起来。像这样一件极具文艺与复古味道的连衣裙，将双层荷叶边塑造娃娃领效果，甜美中透出怀旧的情调。格子图案的加入渲染英伦气息，让身上气质更显优雅干练。腰部的抽绳系带，让裙摆更显蓬松效果，展现淑女的一面。'}
    {'content': '类型#上衣*版型#宽松*颜色#黄色*风格#简约*图案#条纹*图案#蝴蝶结*衣样式#衬衫*衣袖型#喇叭袖*衣门襟#系带', 'summary': '一款简约而不简单的衬衫，宽松合身的款式穿着舒适又不挑身材，看习惯了各种白条或者蓝白条纹的衬衫，暖黄色的条纹是不是给你眼前一亮的感觉呢？袖口小喇叭袖与领口上的蝴蝶结系带相呼应，满满的甜美少女感，穿起来也很减龄。'}
    {'content': '类型#裙*风格#复古*风格#青春*图案#复古*图案#线条*图案#刺绣*裙长#连衣裙*裙领型#翻领*裙款式#收腰', 'summary': '这一款连衣裙看起来公主风十足，翻领的线条流畅，显得整个人很有气质，而领部上面的精致刺绣散发着甜美的气息，有着青春减龄的效果。收腰放摆的廓<UNK>裙摆看起来很蓬松，轻松藏肉。复古的提花面料带来优异的质感，更加分哦。'}
    {'content': '类型#裙*风格#潮*风格#性感*裙型#a字*裙型#鱼尾裙*裙长#连衣裙*裙领型#立领*裙款式#钉珠*裙款式#木耳边', 'summary': 'a字型轮廓的一条连衣裙，不显身材的设计，还不挑人穿，无论你是高个子，还是小个子，都可以轻松的驾驭，让你轻松展现魅惑的女人味。木耳花边的设计，显露穿着甜美感，立领的领口，修饰脸型，显脸小的视觉效果。鱼尾的裙摆，是个性感的设计。钉珠的点缀，增添服装的层次，与潮流感。'}



```python
import json
import random

# 加载JSON文件
file_path = 'data/AdvertiseGen/train.json'  # JSON文件的路径

data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 检查数据长度，如果小于10，则取全部数据
sample_size = min(10, len(data))

# 随机选择10条或者全部数据（如果数据少于10条）
sample_data = random.sample(data, sample_size)

# 打印选中的记录
for item in sample_data:
    print(item)

```

    {'content': '类型#裙*版型#显瘦*风格#复古*风格#性感*图案#复古*图案#波点*图案#线条*图案#印花*裙长#连衣裙*裙款式#抽褶*裙款式#收腰', 'summary': '风靡于上个世纪60年代的波点元素，随着近两年复古风的回潮，再次成为时尚圈的宠儿。这款连衣裙采用最为经典的黑底白点印花，带着轻快活泼的艺术气息，打造优雅复古的look。而领口处以及裙摆的三层褶皱则用小波点来呈现，丰富了整体的视觉层次感，尽显优雅大方。v型领口的设计，展露了女性迷人的肩颈线条及锁骨，彰显小性感。修身收腰的剪裁设计，显瘦效果非常棒，轻松打造优雅复古的女神BRAND。'}
    {'content': '类型#裙*材质#蕾丝*材质#雪纺*风格#复古*风格#简约*图案#复古*图案#印花*图案#蕾丝*裙长#连衣裙*裙领型#圆领*裙款式#拼接*裙款式#腰带*裙款式#不规则', 'summary': '这款连衣裙运用<UNK>印花面料以拼接的设计打造复古典雅的风味，不规则的裙摆在雪纺面料的陪衬下更显潇洒个性。蕾丝面料的衣袖微微展露修长双臂，更显女人味。腰带收束腰身，展现玲珑有致的好身材。圆领简约自然，衬托修长的脖颈和俏丽的小脸。'}
    {'content': '类型#上衣*颜色#黑色*风格#复古*图案#复古*图案#刺绣*衣样式#开衫*衣款式#拼接*衣款式#盘扣', 'summary': '开衫选用了炫酷的黑色基调，辅以刺绣的拼接款式衣襟设计，既简洁立体又精致时尚。配以衣襟的盘扣设计复古时尚，打造干练洒脱的穿搭范。'}
    {'content': '类型#裙*版型#显瘦*颜色#杏色*裙型#背带裙*裙款式#纽扣*裙款式#对称', 'summary': '和通常常见的背带裙款式不同，这款背带裙在背带上做了刻意加宽设计，小小改动使得裙身体现出很有<UNK>的绅士风采。特别是可调节金属扣，与浅杏色有良好的对称和互动。加上裙摆的风琴褶设计，做到了藏肉显瘦，甜美娇俏，以及经典与时尚的全方位整体性融合。'}
    {'content': '类型#裙*材质#针织*风格#复古*风格#知性*风格#性感*图案#复古*裙长#长裙*裙领型#圆领', 'summary': '此款长裙采用冰丝针织的手法，展现设计师巧妙的设计理念，且具有极好的舒适感和透气感。经典的圆领领型，极易搭配饰品且不挑捡脸型。后背处采用美背的效果，个性中带着一丝性感。旗袍式的款式，打造复古柔情的知性女子。'}
    {'content': '类型#裤*版型#显瘦*风格#性感', 'summary': '本款运动裤根据人工学原理剪裁制作，加高的腰头，瞬间<UNK>小肚腩，藏肉显瘦。蜜桃臀的设计走线，轻松勾勒性感翘臀。立体紧身的裤型，穿出修长美腿。反光logo的点缀，利于提高安全辨识度。'}
    {'content': '类型#裙*材质#雪纺*风格#清新*裙型#百褶*裙型#直筒裙*裙长#连衣裙*裙领型#v领*裙袖型#喇叭袖*裙款式#拼接*裙款式#波浪*裙款式#木耳边*裙款式#收腰', 'summary': '温柔气质的雪纺连衣裙，清新淡雅的花朵元素点缀裙身，透露着浓浓的美意让人毫无<UNK>。松紧收腰的直筒版型，搭配上过膝的适中裙长，对身材几乎没有太大的限制，人人都可以穿出属于自己的独特味道。木耳边拼接的小v领设计，精美而别致，既美观又可以修饰脸型，很有设计感哦，浪漫的喇叭袖以及百褶波浪裙摆都显得很有女人味，上身更是显得女神范十足~'}
    {'content': '类型#裙*颜色#绿色*风格#清新*图案#刺绣*裙下摆#开叉*裙款式#盘扣', 'summary': '冰绿色调给人清新的视觉感，如意<UNK>装饰散发出古典的气息。后背还有萌态<UNK>的老虎刺绣图案，瞬间就能帮你减龄。开衩裙摆处的手工盘扣点缀，去除了单调乏味感，注入了中式怀旧气息，提升女人的魅力。'}
    {'content': '类型#上衣*版型#宽松*颜色#黑色*颜色#绿色*风格#运动*风格#清新*风格#潮*风格#工装*图案#字母*图案#条纹*图案#文字*图案#拼色*衣样式#外套*衣款式#松紧带*衣款式#连帽', 'summary': '爵纪品牌推出的这款潮流运动宽松型工装连帽外套，采用朴素自然的绿色调渲染，还带有一丝的清新时尚效果。采用黑色条纹装饰，形成拼色元素的样式，个性十足，凸显与众不同的效果。胸前的部位设有黑色字母图案点缀，简洁美观。橡筋松紧下摆，舒适百搭。'}
    {'content': '类型#裤*风格#复古*风格#休闲*图案#复古*裤款式#口袋', 'summary': '休闲西装裤的版型，不会那么的拘谨和正式，日常休闲也是可以穿出门的，当然平时上班也不会觉得那么随意，它介于两者之间，<UNK>较强，实穿的款。左边口袋加入小巧的金属链夹的设计，复古摩登的设计，带着点中性的风格，都市时装感味道十足。经过前后的<UNK>处理，让裤身更贴合人体曲线，整体是特别工整洁净的。'}


# lora微调

## 1. 准备数据集
我们使用 AdvertiseGen 数据集来进行微调。从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 AdvertiseGen 数据集，将解压后的 AdvertiseGen 目录放到本目录的 `/data/` 下, 例如。
> /media/zr/Data/Code/ChatGLM3/finetune_demo/data/AdvertiseGen

接着，运行本代码来切割数据集


```python
import json
from typing import Union
from pathlib import Path


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']},
                                                {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)


convert_adgen('data/AdvertiseGen', 'data/AdvertiseGen_fix')
```

## 2. 使用命令行开始微调,我们使用 lora 进行微调
接着，我们仅需要将配置好的参数以命令行的形式传参给程序，就可以使用命令行进行高效微调，这里将 `/media/zr/Data/Code/ChatGLM3/venv/bin/python3` 换成你的 python3 的绝对路径以保证正常运行。

### 设置学术加速


```python
!source /etc/network_turbo
```

    设置成功



```python
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
```


```python
cat /etc/network_turbo
```

    export no_proxy=localhost,127.0.0.1
    export http_proxy=http://192.168.126.12:12798 && export https_proxy=http://192.168.126.12:12798
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    echo 设置成功



```python
!unset http_proxy && unset https_proxy
```


```python
### 使用Model Scope库下载模型
```


```python
pip install modelscope
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Collecting modelscope
      Using cached http://mirrors.aliyun.com/pypi/packages/83/9f/5a7802670bbd13e69d110032ba8aab0264dc42d82b4b7e87f4396647c0ae/modelscope-1.13.3-py3-none-any.whl (5.7 MB)
    Requirement already satisfied: requests>=2.25 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (2.31.0)
    Collecting einops
      Downloading http://mirrors.aliyun.com/pypi/packages/29/0b/2d1c0ebfd092e25935b86509a9a817159212d82aa43d7fb07eca4eeff2c2/einops-0.7.0-py3-none-any.whl (44 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.6/44.6 kB[0m [31m328.9 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: numpy in ./miniconda3/lib/python3.10/site-packages (from modelscope) (1.26.3)
    Requirement already satisfied: tqdm>=4.64.0 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (4.64.1)
    Collecting simplejson>=3.3.0
      Using cached http://mirrors.aliyun.com/pypi/packages/cb/b6/ed513a0adc3e2c9654864ffb68266dcab5720d5653428d690e7e4fb32a6c/simplejson-3.19.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (137 kB)
    Requirement already satisfied: pyyaml in ./miniconda3/lib/python3.10/site-packages (from modelscope) (6.0.1)
    Collecting oss2
      Using cached http://mirrors.aliyun.com/pypi/packages/d5/63/b6c355af7f04a8a1d5759fa6fc47539e25ef8e6f2745372a242fdadcac65/oss2-2.18.4.tar.gz (278 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting addict
      Downloading http://mirrors.aliyun.com/pypi/packages/6a/00/b08f23b7d7e1e14ce01419a467b583edbb93c6cdb8654e54a9cc579cd61f/addict-2.4.0-py3-none-any.whl (3.8 kB)
    Collecting gast>=0.2.2
      Downloading http://mirrors.aliyun.com/pypi/packages/fa/39/5aae571e5a5f4de9c3445dae08a530498e5c53b0e74410eeeb0991c79047/gast-0.5.4-py3-none-any.whl (19 kB)
    Requirement already satisfied: filelock>=3.3.0 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (3.13.1)
    Requirement already satisfied: Pillow>=6.2.0 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (10.2.0)
    Collecting yapf
      Downloading http://mirrors.aliyun.com/pypi/packages/66/c9/d4b03b2490107f13ebd68fe9496d41ae41a7de6275ead56d0d4621b11ffd/yapf-0.40.2-py3-none-any.whl (254 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m254.7/254.7 kB[0m [31m367.3 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: huggingface-hub in ./miniconda3/lib/python3.10/site-packages (from modelscope) (0.22.2)
    Collecting sortedcontainers>=1.5.9
      Downloading http://mirrors.aliyun.com/pypi/packages/32/46/9cb0e58b2deb7f82b84065f37f3bffeb12413f947f9388e4cac22c4621ce/sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
    Requirement already satisfied: urllib3>=1.26 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (1.26.13)
    Requirement already satisfied: pandas in ./miniconda3/lib/python3.10/site-packages (from modelscope) (2.2.1)
    Requirement already satisfied: python-dateutil>=2.1 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (2.8.2)
    Requirement already satisfied: pyarrow!=9.0.0,>=6.0.0 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (15.0.2)
    Requirement already satisfied: datasets>=2.14.5 in ./miniconda3/lib/python3.10/site-packages (from modelscope) (2.18.0)
    Requirement already satisfied: setuptools in ./miniconda3/lib/python3.10/site-packages (from modelscope) (65.5.0)
    Collecting scipy
      Downloading http://mirrors.aliyun.com/pypi/packages/f5/aa/8e6071a5e4dca4ec68b5b22e4991ee74c59c5d372112b9c236ec1faff57d/scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m38.4/38.4 MB[0m [31m355.3 kB/s[0m eta [36m0:00:00[0m00:01[0m00:03[0m
    [?25hRequirement already satisfied: attrs in ./miniconda3/lib/python3.10/site-packages (from modelscope) (23.2.0)
    Requirement already satisfied: multiprocess in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (0.70.16)
    Requirement already satisfied: pyarrow-hotfix in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (0.6)
    Requirement already satisfied: xxhash in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (3.4.1)
    Requirement already satisfied: fsspec[http]<=2024.2.0,>=2023.1.0 in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (2023.12.2)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (0.3.8)
    Requirement already satisfied: aiohttp in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (3.9.3)
    Requirement already satisfied: packaging in ./miniconda3/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (23.2)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in ./miniconda3/lib/python3.10/site-packages (from huggingface-hub->modelscope) (4.9.0)
    Requirement already satisfied: six>=1.5 in ./miniconda3/lib/python3.10/site-packages (from python-dateutil>=2.1->modelscope) (1.16.0)
    Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/lib/python3.10/site-packages (from requests>=2.25->modelscope) (3.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in ./miniconda3/lib/python3.10/site-packages (from requests>=2.25->modelscope) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/lib/python3.10/site-packages (from requests>=2.25->modelscope) (2022.12.7)
    Collecting crcmod>=1.7
      Downloading http://mirrors.aliyun.com/pypi/packages/6b/b0/e595ce2a2527e169c3bcd6c33d2473c1918e0b7f6826a043ca1245dd4e5b/crcmod-1.7.tar.gz (89 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.7/89.7 kB[0m [31m338.1 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting pycryptodome>=3.4.7
      Downloading http://mirrors.aliyun.com/pypi/packages/af/20/5f29ec45462360e7f61e8688af9fe4a0afae057edfabdada662e11bf97e7/pycryptodome-3.20.0-cp35-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.1/2.1 MB[0m [31m348.5 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting aliyun-python-sdk-kms>=2.4.1
      Downloading http://mirrors.aliyun.com/pypi/packages/3d/ea/d88e08bfc4a0aee0111f1f24c98b19107bc6783441e7e944907c77b2243d/aliyun_python_sdk_kms-2.16.2-py2.py3-none-any.whl (94 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m94.0/94.0 kB[0m [31m319.4 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting aliyun-python-sdk-core>=2.13.12
      Downloading http://mirrors.aliyun.com/pypi/packages/cf/0f/c191007d4a0c068725009489d7f928614151da938598b875568a6323cff2/aliyun-python-sdk-core-2.15.0.tar.gz (443 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m443.1/443.1 kB[0m [31m350.3 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: pytz>=2020.1 in ./miniconda3/lib/python3.10/site-packages (from pandas->modelscope) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in ./miniconda3/lib/python3.10/site-packages (from pandas->modelscope) (2024.1)
    Requirement already satisfied: platformdirs>=3.5.1 in ./miniconda3/lib/python3.10/site-packages (from yapf->modelscope) (4.1.0)
    Requirement already satisfied: tomli>=2.0.1 in ./miniconda3/lib/python3.10/site-packages (from yapf->modelscope) (2.0.1)
    Collecting importlib-metadata>=6.6.0
      Downloading http://mirrors.aliyun.com/pypi/packages/2d/0a/679461c511447ffaf176567d5c496d1de27cbe34a87df6677d7171b2fbd4/importlib_metadata-7.1.0-py3-none-any.whl (24 kB)
    Collecting jmespath<1.0.0,>=0.9.3
      Downloading http://mirrors.aliyun.com/pypi/packages/07/cb/5f001272b6faeb23c1c9e0acc04d48eaaf5c862c17709d20e3469c6e0139/jmespath-0.10.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: cryptography>=2.6.0 in ./miniconda3/lib/python3.10/site-packages (from aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (38.0.1)
    Requirement already satisfied: aiosignal>=1.1.2 in ./miniconda3/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (1.3.1)
    Requirement already satisfied: frozenlist>=1.1.1 in ./miniconda3/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in ./miniconda3/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in ./miniconda3/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in ./miniconda3/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (4.0.3)
    Collecting zipp>=0.5
      Downloading http://mirrors.aliyun.com/pypi/packages/c2/0a/ba9d0ee9536d3ef73a3448e931776e658b36f128d344e175bc32b092a8bf/zipp-3.18.1-py3-none-any.whl (8.2 kB)
    Requirement already satisfied: cffi>=1.12 in ./miniconda3/lib/python3.10/site-packages (from cryptography>=2.6.0->aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (1.15.1)
    Requirement already satisfied: pycparser in ./miniconda3/lib/python3.10/site-packages (from cffi>=1.12->cryptography>=2.6.0->aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (2.21)
    Building wheels for collected packages: oss2, aliyun-python-sdk-core, crcmod
      Building wheel for oss2 (setup.py) ... [?25ldone
    [?25h  Created wheel for oss2: filename=oss2-2.18.4-py3-none-any.whl size=115944 sha256=fae922e9fb19c419370d0968c2c142df373430517634453c7145f00568a18335
      Stored in directory: /root/.cache/pip/wheels/d2/88/c8/dc5ade82f4a9607d9982bfbf37754bcb7ef3fa3444646a59f9
      Building wheel for aliyun-python-sdk-core (setup.py) ... [?25ldone
    [?25h  Created wheel for aliyun-python-sdk-core: filename=aliyun_python_sdk_core-2.15.0-py3-none-any.whl size=535314 sha256=6bae9696ec499f29f83888f14013771b24f5b5faf3ac81a45dd021aeb0f1ed26
      Stored in directory: /root/.cache/pip/wheels/4b/c1/b3/beefcc3188ae130e1288f0df7fb6afdbf5b13e90736df5db3d
      Building wheel for crcmod (setup.py) ... [?25ldone
    [?25h  Created wheel for crcmod: filename=crcmod-1.7-cp310-cp310-linux_x86_64.whl size=23521 sha256=e913d98cea73d28cf6c1fb32301bae29f3895978ab18348873e229725be7464e
      Stored in directory: /root/.cache/pip/wheels/91/b3/b0/28f3c022098be1077b56b2084361ce3dabf549f1b56ead0f11
    Successfully built oss2 aliyun-python-sdk-core crcmod
    Installing collected packages: sortedcontainers, crcmod, addict, zipp, simplejson, scipy, pycryptodome, jmespath, gast, einops, importlib-metadata, yapf, aliyun-python-sdk-core, aliyun-python-sdk-kms, oss2, modelscope
    Successfully installed addict-2.4.0 aliyun-python-sdk-core-2.15.0 aliyun-python-sdk-kms-2.16.2 crcmod-1.7 einops-0.7.0 gast-0.5.4 importlib-metadata-7.1.0 jmespath-0.10.0 modelscope-1.13.3 oss2-2.18.4 pycryptodome-3.20.0 scipy-1.12.0 simplejson-3.19.2 sortedcontainers-2.4.0 yapf-0.40.2 zipp-3.18.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
from modelscope.models import Model
model = Model.from_pretrained('ZhipuAI/chatglm3-6b')
```

    2024-04-03 01:14:56,139 - modelscope - WARNING - Model revision not specified, use revision: v1.0.2
    Downloading: 100%|██████████| 1.29k/1.29k [00:00<00:00, 1.14MB/s]
    Downloading: 100%|██████████| 37.0/37.0 [00:00<00:00, 38.3kB/s]
    Downloading: 100%|██████████| 2.28k/2.28k [00:00<00:00, 1.46MB/s]
    Downloading: 100%|██████████| 4.04k/4.04k [00:00<00:00, 358kB/s]
    Downloading: 100%|██████████| 54.3k/54.3k [00:00<00:00, 617kB/s]
    Downloading: 100%|█████████▉| 1.70G/1.70G [00:35<00:00, 51.1MB/s]
    Downloading: 100%|█████████▉| 1.83G/1.83G [00:43<00:00, 45.5MB/s]
    Downloading: 100%|█████████▉| 1.80G/1.80G [00:44<00:00, 43.3MB/s]
    Downloading: 100%|█████████▉| 1.69G/1.69G [00:30<00:00, 59.9MB/s]
    Downloading: 100%|█████████▉| 1.83G/1.83G [00:43<00:00, 45.4MB/s]
    Downloading: 100%|█████████▉| 1.80G/1.80G [00:36<00:00, 52.2MB/s]
    Downloading: 100%|█████████▉| 0.98G/0.98G [00:22<00:00, 47.5MB/s]
    Downloading: 100%|██████████| 20.0k/20.0k [00:00<00:00, 470kB/s]
    Downloading: 100%|██████████| 14.3k/14.3k [00:00<00:00, 335kB/s]
    Downloading: 100%|██████████| 4.37k/4.37k [00:00<00:00, 382kB/s]
    Downloading: 100%|██████████| 11.0k/11.0k [00:00<00:00, 8.14MB/s]
    Downloading: 100%|██████████| 995k/995k [00:00<00:00, 3.05MB/s]
    Downloading: 100%|██████████| 244/244 [00:00<00:00, 247kB/s]
    2024-04-03 01:19:47,203 - modelscope - INFO - initialize model from /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b


    The repository for /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co//root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b.
    You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
    
    Do you wish to run the custom code? [y/N]  y
    The repository for /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co//root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b.
    You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
    
    Do you wish to run the custom code? [y/N]  y



    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]


    /root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()


### 使用Hugging Face库下载模型(无法连接，下载失败)


```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
```


    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connection.py:174, in HTTPConnection._new_conn(self)
        173 try:
    --> 174     conn = connection.create_connection(
        175         (self._dns_host, self.port), self.timeout, **extra_kw
        176     )
        178 except SocketTimeout:


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/util/connection.py:95, in create_connection(address, timeout, source_address, socket_options)
         94 if err is not None:
    ---> 95     raise err
         97 raise socket.error("getaddrinfo returns an empty list")


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/util/connection.py:85, in create_connection(address, timeout, source_address, socket_options)
         84     sock.bind(source_address)
    ---> 85 sock.connect(sa)
         86 return sock


    TimeoutError: timed out

    
    During handling of the above exception, another exception occurred:


    ConnectTimeoutError                       Traceback (most recent call last)

    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connectionpool.py:703, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        702 # Make the request on the httplib connection object.
    --> 703 httplib_response = self._make_request(
        704     conn,
        705     method,
        706     url,
        707     timeout=timeout_obj,
        708     body=body,
        709     headers=headers,
        710     chunked=chunked,
        711 )
        713 # If we're going to release the connection in ``finally:``, then
        714 # the response doesn't need to know about the connection. Otherwise
        715 # it will also try to release it and we'll have a double-release
        716 # mess.


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connectionpool.py:386, in HTTPConnectionPool._make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        385 try:
    --> 386     self._validate_conn(conn)
        387 except (SocketTimeout, BaseSSLError) as e:
        388     # Py2 raises this as a BaseSSLError, Py3 raises it as socket timeout.


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connectionpool.py:1042, in HTTPSConnectionPool._validate_conn(self, conn)
       1041 if not getattr(conn, "sock", None):  # AppEngine might not have  `.sock`
    -> 1042     conn.connect()
       1044 if not conn.is_verified:


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connection.py:358, in HTTPSConnection.connect(self)
        356 def connect(self):
        357     # Add certificate verification
    --> 358     self.sock = conn = self._new_conn()
        359     hostname = self.host


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connection.py:179, in HTTPConnection._new_conn(self)
        178 except SocketTimeout:
    --> 179     raise ConnectTimeoutError(
        180         self,
        181         "Connection to %s timed out. (connect timeout=%s)"
        182         % (self.host, self.timeout),
        183     )
        185 except SocketError as e:


    ConnectTimeoutError: (<urllib3.connection.HTTPSConnection object at 0x7f1aaff30df0>, 'Connection to huggingface.co timed out. (connect timeout=10)')

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    File ~/miniconda3/lib/python3.10/site-packages/requests/adapters.py:486, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
        485 try:
    --> 486     resp = conn.urlopen(
        487         method=request.method,
        488         url=url,
        489         body=request.body,
        490         headers=request.headers,
        491         redirect=False,
        492         assert_same_host=False,
        493         preload_content=False,
        494         decode_content=False,
        495         retries=self.max_retries,
        496         timeout=timeout,
        497         chunked=chunked,
        498     )
        500 except (ProtocolError, OSError) as err:


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/connectionpool.py:787, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        785     e = ProtocolError("Connection aborted.", e)
    --> 787 retries = retries.increment(
        788     method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2]
        789 )
        790 retries.sleep()


    File ~/miniconda3/lib/python3.10/site-packages/urllib3/util/retry.py:592, in Retry.increment(self, method, url, response, error, _pool, _stacktrace)
        591 if new_retry.is_exhausted():
    --> 592     raise MaxRetryError(_pool, url, error or ResponseError(cause))
        594 log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)


    MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /THUDM/chatglm3-6b/resolve/main/config.json (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x7f1aaff30df0>, 'Connection to huggingface.co timed out. (connect timeout=10)'))

    
    During handling of the above exception, another exception occurred:


    ConnectTimeout                            Traceback (most recent call last)

    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:1261, in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, local_dir_use_symlinks, user_agent, force_download, force_filename, proxies, etag_timeout, resume_download, token, local_files_only, headers, legacy_cache_layout, endpoint)
       1260 try:
    -> 1261     metadata = get_hf_file_metadata(
       1262         url=url,
       1263         token=token,
       1264         proxies=proxies,
       1265         timeout=etag_timeout,
       1266         library_name=library_name,
       1267         library_version=library_version,
       1268         user_agent=user_agent,
       1269     )
       1270 except EntryNotFoundError as http_error:
       1271     # Cache the non-existence of the file and raise


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py:119, in validate_hf_hub_args.<locals>._inner_fn(*args, **kwargs)
        117     kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
    --> 119 return fn(*args, **kwargs)


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:1674, in get_hf_file_metadata(url, token, proxies, timeout, library_name, library_version, user_agent, headers)
       1673 # Retrieve metadata
    -> 1674 r = _request_wrapper(
       1675     method="HEAD",
       1676     url=url,
       1677     headers=headers,
       1678     allow_redirects=False,
       1679     follow_relative_redirects=True,
       1680     proxies=proxies,
       1681     timeout=timeout,
       1682 )
       1683 hf_raise_for_status(r)


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:369, in _request_wrapper(method, url, follow_relative_redirects, **params)
        368 if follow_relative_redirects:
    --> 369     response = _request_wrapper(
        370         method=method,
        371         url=url,
        372         follow_relative_redirects=False,
        373         **params,
        374     )
        376     # If redirection, we redirect only relative paths.
        377     # This is useful in case of a renamed repository.


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:392, in _request_wrapper(method, url, follow_relative_redirects, **params)
        391 # Perform request and return if status_code is not in the retry list.
    --> 392 response = get_session().request(method=method, url=url, **params)
        393 hf_raise_for_status(response)


    File ~/miniconda3/lib/python3.10/site-packages/requests/sessions.py:589, in Session.request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        588 send_kwargs.update(settings)
    --> 589 resp = self.send(prep, **send_kwargs)
        591 return resp


    File ~/miniconda3/lib/python3.10/site-packages/requests/sessions.py:703, in Session.send(self, request, **kwargs)
        702 # Send the request
    --> 703 r = adapter.send(request, **kwargs)
        705 # Total elapsed time of the request (approximately)


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/utils/_http.py:68, in UniqueRequestIdAdapter.send(self, request, *args, **kwargs)
         67 try:
    ---> 68     return super().send(request, *args, **kwargs)
         69 except requests.RequestException as e:


    File ~/miniconda3/lib/python3.10/site-packages/requests/adapters.py:507, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
        506     if not isinstance(e.reason, NewConnectionError):
    --> 507         raise ConnectTimeout(e, request=request)
        509 if isinstance(e.reason, ResponseError):


    ConnectTimeout: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /THUDM/chatglm3-6b/resolve/main/config.json (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x7f1aaff30df0>, 'Connection to huggingface.co timed out. (connect timeout=10)'))"), '(Request ID: 0fa5114f-adbf-4278-b4b4-bb05c9359dc7)')

    
    The above exception was the direct cause of the following exception:


    LocalEntryNotFoundError                   Traceback (most recent call last)

    File ~/miniconda3/lib/python3.10/site-packages/transformers/utils/hub.py:398, in cached_file(path_or_repo_id, filename, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
        396 try:
        397     # Load from URL or cache if already cached
    --> 398     resolved_file = hf_hub_download(
        399         path_or_repo_id,
        400         filename,
        401         subfolder=None if len(subfolder) == 0 else subfolder,
        402         repo_type=repo_type,
        403         revision=revision,
        404         cache_dir=cache_dir,
        405         user_agent=user_agent,
        406         force_download=force_download,
        407         proxies=proxies,
        408         resume_download=resume_download,
        409         token=token,
        410         local_files_only=local_files_only,
        411     )
        412 except GatedRepoError as e:


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py:119, in validate_hf_hub_args.<locals>._inner_fn(*args, **kwargs)
        117     kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
    --> 119 return fn(*args, **kwargs)


    File ~/miniconda3/lib/python3.10/site-packages/huggingface_hub/file_download.py:1406, in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, local_dir_use_symlinks, user_agent, force_download, force_filename, proxies, etag_timeout, resume_download, token, local_files_only, headers, legacy_cache_layout, endpoint)
       1404     else:
       1405         # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
    -> 1406         raise LocalEntryNotFoundError(
       1407             "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
       1408             " in the local cache. Please check your connection and try again or make sure your Internet connection"
       1409             " is on."
       1410         ) from head_call_error
       1412 # From now on, etag and commit_hash are not None.


    LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.

    
    The above exception was the direct cause of the following exception:


    OSError                                   Traceback (most recent call last)

    Cell In[3], line 2
          1 from transformers import AutoTokenizer, AutoModel
    ----> 2 tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
          3 model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()


    File ~/miniconda3/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py:794, in AutoTokenizer.from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
        792 if config_tokenizer_class is None:
        793     if not isinstance(config, PretrainedConfig):
    --> 794         config = AutoConfig.from_pretrained(
        795             pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        796         )
        797     config_tokenizer_class = config.tokenizer_class
        798     if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:


    File ~/miniconda3/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py:1138, in AutoConfig.from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
       1135 trust_remote_code = kwargs.pop("trust_remote_code", None)
       1136 code_revision = kwargs.pop("code_revision", None)
    -> 1138 config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
       1139 has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
       1140 has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING


    File ~/miniconda3/lib/python3.10/site-packages/transformers/configuration_utils.py:631, in PretrainedConfig.get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
        629 original_kwargs = copy.deepcopy(kwargs)
        630 # Get config dict associated with the base config file
    --> 631 config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        632 if "_commit_hash" in config_dict:
        633     original_kwargs["_commit_hash"] = config_dict["_commit_hash"]


    File ~/miniconda3/lib/python3.10/site-packages/transformers/configuration_utils.py:686, in PretrainedConfig._get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
        682 configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)
        684 try:
        685     # Load from local folder or from cache or download from model Hub and cache
    --> 686     resolved_config_file = cached_file(
        687         pretrained_model_name_or_path,
        688         configuration_file,
        689         cache_dir=cache_dir,
        690         force_download=force_download,
        691         proxies=proxies,
        692         resume_download=resume_download,
        693         local_files_only=local_files_only,
        694         token=token,
        695         user_agent=user_agent,
        696         revision=revision,
        697         subfolder=subfolder,
        698         _commit_hash=commit_hash,
        699     )
        700     commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        701 except EnvironmentError:
        702     # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
        703     # the original exception.


    File ~/miniconda3/lib/python3.10/site-packages/transformers/utils/hub.py:441, in cached_file(path_or_repo_id, filename, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
        435     if (
        436         resolved_file is not None
        437         or not _raise_exceptions_for_missing_entries
        438         or not _raise_exceptions_for_connection_errors
        439     ):
        440         return resolved_file
    --> 441     raise EnvironmentError(
        442         f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
        443         f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
        444         f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
        445         " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        446     ) from e
        447 except EntryNotFoundError as e:
        448     if not _raise_exceptions_for_missing_entries:


    OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like THUDM/chatglm3-6b is not the path to a directory containing a file named config.json.
    Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.


## 找到模型下载目录


```python
ll /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b
```

    total 12195720
    -rw------- 1 root       4133 Apr  3 01:14 MODEL_LICENSE
    -rw------- 1 root       4478 Apr  3 01:19 README.md
    -rw------- 1 root       1317 Apr  3 01:14 config.json
    -rw------- 1 root         37 Apr  3 01:14 configuration.json
    -rw------- 1 root       2332 Apr  3 01:14 configuration_chatglm.py
    -rw------- 1 root      55596 Apr  3 01:14 modeling_chatglm.py
    -rw------- 1 root 1827781090 Apr  3 01:15 pytorch_model-00001-of-00007.bin
    -rw------- 1 root 1968299480 Apr  3 01:16 pytorch_model-00002-of-00007.bin
    -rw------- 1 root 1927415036 Apr  3 01:17 pytorch_model-00003-of-00007.bin
    -rw------- 1 root 1815225998 Apr  3 01:17 pytorch_model-00004-of-00007.bin
    -rw------- 1 root 1968299544 Apr  3 01:18 pytorch_model-00005-of-00007.bin
    -rw------- 1 root 1927415036 Apr  3 01:19 pytorch_model-00006-of-00007.bin
    -rw------- 1 root 1052808542 Apr  3 01:19 pytorch_model-00007-of-00007.bin
    -rw------- 1 root      20437 Apr  3 01:19 pytorch_model.bin.index.json
    -rw------- 1 root      14692 Apr  3 01:19 quantization.py
    -rw------- 1 root      11279 Apr  3 01:19 tokenization_chatglm.py
    -rw------- 1 root    1018370 Apr  3 01:19 tokenizer.model
    -rw------- 1 root        244 Apr  3 01:19 tokenizer_config.json


## 找到Python3位置


```python
!which python3
```

    /root/miniconda3/bin/python3


## 开始微调


```python
cat configs/lora.yaml
```

    data_config:
      train_file: train.json
      val_file: dev.json
      test_file: dev.json
      num_proc: 16
    max_input_length: 256
    max_output_length: 512
    training_args:
      # see `transformers.Seq2SeqTrainingArguments`
      output_dir: ./output
      max_steps: 3000
      # needed to be fit for the dataset
      learning_rate: 5e-5
      # settings for data loading
      per_device_train_batch_size: 4
      dataloader_num_workers: 16
      remove_unused_columns: false
      # settings for saving checkpoints
      save_strategy: steps
      save_steps: 2000
      # settings for logging
      log_level: info
      logging_strategy: steps
      logging_steps: 10
      # settings for evaluation
      per_device_eval_batch_size: 16
      evaluation_strategy: steps
      eval_steps: 500
      # settings for optimizer
      # adam_epsilon: 1e-6
      # uncomment the following line to detect nan or inf values
      # debug: underflow_overflow
      predict_with_generate: true
      # see `transformers.GenerationConfig`
      generation_config:
        max_new_tokens: 512
      # set your absolute deepspeed path here
      #deepspeed: ds_zero_2.json
      # set to true if train with cpu.
      use_cpu: false
    peft_config:
      peft_type: LORA
      task_type: CAUSAL_LM
      r: 8
      lora_alpha: 32
      lora_dropout: 0.1



```python
!/root/miniconda3/bin/python3 finetune_hf.py  data/AdvertiseGen_fix  /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b  configs/lora.yaml
```

    Loading checkpoint shards:   0%|                          | 0/7 [00:00<?, ?it/s]/root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    Loading checkpoint shards: 100%|██████████████████| 7/7 [00:04<00:00,  1.45it/s]
    trainable params: 1,949,696 || all params: 6,245,533,696 || trainable%: 0.031217444255383614
    --> Model
    
    --> model has 1.949696M params
    
    train_dataset: Dataset({
        features: ['input_ids', 'labels'],
        num_rows: 114599
    })
    Map (num_proc=16): 100%|███████████| 1070/1070 [00:00<00:00, 1171.33 examples/s]
    val_dataset: Dataset({
        features: ['input_ids', 'output_ids'],
        num_rows: 1070
    })
    test_dataset: Dataset({
        features: ['input_ids', 'output_ids'],
        num_rows: 1070
    })
    --> Sanity check
               '[gMASK]': 64790 -> -100
                   'sop': 64792 -> -100
              '<|user|>': 64795 -> -100
                      '': 30910 -> -100
                    '\n': 13 -> -100
                      '': 30910 -> -100
                    '类型': 33467 -> -100
                     '#': 31010 -> -100
                     '裤': 56532 -> -100
                     '*': 30998 -> -100
                     '版': 55090 -> -100
                     '型': 54888 -> -100
                     '#': 31010 -> -100
                    '宽松': 40833 -> -100
                     '*': 30998 -> -100
                    '风格': 32799 -> -100
                     '#': 31010 -> -100
                    '性感': 40589 -> -100
                     '*': 30998 -> -100
                    '图案': 37505 -> -100
                     '#': 31010 -> -100
                    '线条': 37216 -> -100
                     '*': 30998 -> -100
                     '裤': 56532 -> -100
                     '型': 54888 -> -100
                     '#': 31010 -> -100
                     '阔': 56529 -> -100
                     '腿': 56158 -> -100
                     '裤': 56532 -> -100
         '<|assistant|>': 64796 -> -100
                      '': 30910 -> 30910
                    '\n': 13 -> 13
                      '': 30910 -> 30910
                    '宽松': 40833 -> 40833
                     '的': 54530 -> 54530
                     '阔': 56529 -> 56529
                     '腿': 56158 -> 56158
                     '裤': 56532 -> 56532
                     '这': 54551 -> 54551
                    '两年': 33808 -> 33808
                    '真的': 32041 -> 32041
                     '吸': 55360 -> 55360
                     '粉': 55486 -> 55486
                    '不少': 32138 -> 32138
                     '，': 31123 -> 31123
                    '明星': 32943 -> 32943
                    '时尚': 33481 -> 33481
                     '达': 54880 -> 54880
                    '人的': 31664 -> 31664
                    '心头': 46565 -> 46565
                     '爱': 54799 -> 54799
                     '。': 31155 -> 31155
                    '毕竟': 33051 -> 33051
                     '好': 54591 -> 54591
                     '穿': 55432 -> 55432
                    '时尚': 33481 -> 33481
                     '，': 31123 -> 31123
                     '谁': 55622 -> 55622
                    '都能': 32904 -> 32904
                     '穿': 55432 -> 55432
                     '出': 54557 -> 54557
                     '腿': 56158 -> 56158
                     '长': 54625 -> 54625
                     '2': 30943 -> 30943
                     '米': 55055 -> 55055
                   '的效果': 35590 -> 35590
                    '宽松': 40833 -> 40833
                     '的': 54530 -> 54530
                     '裤': 56532 -> 56532
                     '腿': 56158 -> 56158
                     '，': 31123 -> 31123
                   '当然是': 48466 -> 48466
                     '遮': 57148 -> 57148
                     '肉': 55343 -> 55343
                     '小': 54603 -> 54603
                    '能手': 49355 -> 49355
                     '啊': 55674 -> 55674
                     '。': 31155 -> 31155
                    '上身': 51605 -> 51605
                     '随': 55119 -> 55119
                     '性': 54642 -> 54642
                    '自然': 31799 -> 31799
                     '不': 54535 -> 54535
                     '拘': 57036 -> 57036
                     '束': 55625 -> 55625
                     '，': 31123 -> 31123
                    '面料': 46839 -> 46839
                     '亲': 55113 -> 55113
                     '肤': 56089 -> 56089
                    '舒适': 33894 -> 33894
                     '贴': 55778 -> 55778
                    '身体': 31902 -> 31902
                     '验': 55017 -> 55017
                     '感': 54706 -> 54706
                     '棒': 56382 -> 56382
                     '棒': 56382 -> 56382
                     '哒': 59230 -> 59230
                     '。': 31155 -> 31155
                     '系': 54712 -> 54712
                     '带': 54882 -> 54882
                    '部分': 31726 -> 31726
                    '增加': 31917 -> 31917
                    '设计': 31735 -> 31735
                    '看点': 45032 -> 45032
                     '，': 31123 -> 31123
                     '还': 54656 -> 54656
                     '让': 54772 -> 54772
                    '单品': 46539 -> 46539
                   '的设计': 34481 -> 34481
                     '感': 54706 -> 54706
                    '更强': 43084 -> 43084
                     '。': 31155 -> 31155
                    '腿部': 46799 -> 46799
                    '线条': 37216 -> 37216
                     '若': 55351 -> 55351
                     '隐': 55733 -> 55733
                     '若': 55351 -> 55351
                     '现': 54600 -> 54600
                     '的': 54530 -> 54530
                     '，': 31123 -> 31123
                    '性感': 40589 -> 40589
                     '撩': 58521 -> 58521
                     '人': 54533 -> 54533
                     '。': 31155 -> 31155
                    '颜色': 33692 -> 33692
                     '敲': 57004 -> 57004
                    '温柔': 34678 -> 34678
                     '的': 54530 -> 54530
                     '，': 31123 -> 31123
                     '与': 54619 -> 54619
                    '裤子': 44722 -> 44722
                    '本身': 32754 -> 32754
                     '所': 54626 -> 54626
                    '呈现': 33169 -> 33169
                   '的风格': 48084 -> 48084
                    '有点': 33149 -> 33149
                     '反': 54955 -> 54955
                     '差': 55342 -> 55342
                     '萌': 56842 -> 56842
                     '。': 31155 -> 31155
                      '': 2 -> 2
    You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it).Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model.
    /root/miniconda3/lib/python3.10/site-packages/accelerate/accelerator.py:432: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
    dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False, even_batches=True, use_seedable_sampler=True)
      warnings.warn(
    max_steps is given, it will override any value given in num_train_epochs
    ***** Running training *****
      Num examples = 114,599
      Num Epochs = 1
      Instantaneous batch size per device = 4
      Total train batch size (w. parallel, distributed & accumulation) = 4
      Gradient Accumulation steps = 1
      Total optimization steps = 3,000
      Number of trainable parameters = 1,949,696
      0%|                                                  | 0/3000 [00:00<?, ?it/s]/root/miniconda3/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
      warnings.warn(
    {'loss': 4.8305, 'grad_norm': 2.2845351696014404, 'learning_rate': 4.9833333333333336e-05, 'epoch': 0.0}
    {'loss': 4.6012, 'grad_norm': 3.208742380142212, 'learning_rate': 4.966666666666667e-05, 'epoch': 0.0}
    {'loss': 4.4859, 'grad_norm': 3.011120557785034, 'learning_rate': 4.9500000000000004e-05, 'epoch': 0.0}
    {'loss': 4.1201, 'grad_norm': 3.434319496154785, 'learning_rate': 4.933333333333334e-05, 'epoch': 0.0}
    {'loss': 4.1174, 'grad_norm': 2.7801601886749268, 'learning_rate': 4.9166666666666665e-05, 'epoch': 0.0}
    {'loss': 3.8691, 'grad_norm': 2.981532096862793, 'learning_rate': 4.9e-05, 'epoch': 0.0}
    {'loss': 3.84, 'grad_norm': 2.9055285453796387, 'learning_rate': 4.883333333333334e-05, 'epoch': 0.0}
    {'loss': 3.7453, 'grad_norm': 2.960925817489624, 'learning_rate': 4.866666666666667e-05, 'epoch': 0.0}
    {'loss': 3.6342, 'grad_norm': 3.2687976360321045, 'learning_rate': 4.85e-05, 'epoch': 0.0}
    {'loss': 3.7199, 'grad_norm': 3.4140050411224365, 'learning_rate': 4.8333333333333334e-05, 'epoch': 0.0}
    {'loss': 3.6697, 'grad_norm': 3.646791934967041, 'learning_rate': 4.8166666666666674e-05, 'epoch': 0.0}
    {'loss': 3.8459, 'grad_norm': 3.8958590030670166, 'learning_rate': 4.8e-05, 'epoch': 0.0}
    {'loss': 3.6107, 'grad_norm': 3.533724069595337, 'learning_rate': 4.7833333333333335e-05, 'epoch': 0.0}
    {'loss': 3.7305, 'grad_norm': 4.440822601318359, 'learning_rate': 4.766666666666667e-05, 'epoch': 0.0}
    {'loss': 3.6816, 'grad_norm': 3.676931142807007, 'learning_rate': 4.75e-05, 'epoch': 0.01}
    {'loss': 3.7441, 'grad_norm': 3.9477882385253906, 'learning_rate': 4.7333333333333336e-05, 'epoch': 0.01}
    {'loss': 3.576, 'grad_norm': 4.12568473815918, 'learning_rate': 4.716666666666667e-05, 'epoch': 0.01}
    {'loss': 3.574, 'grad_norm': 4.313591003417969, 'learning_rate': 4.7e-05, 'epoch': 0.01}
    {'loss': 3.5494, 'grad_norm': 4.830262184143066, 'learning_rate': 4.683333333333334e-05, 'epoch': 0.01}
    {'loss': 3.577, 'grad_norm': 4.534295082092285, 'learning_rate': 4.666666666666667e-05, 'epoch': 0.01}
    {'loss': 3.5525, 'grad_norm': 5.008376121520996, 'learning_rate': 4.6500000000000005e-05, 'epoch': 0.01}
    {'loss': 3.6434, 'grad_norm': 4.083359718322754, 'learning_rate': 4.633333333333333e-05, 'epoch': 0.01}
    {'loss': 3.6096, 'grad_norm': 4.787063121795654, 'learning_rate': 4.6166666666666666e-05, 'epoch': 0.01}
    {'loss': 3.51, 'grad_norm': 4.525018215179443, 'learning_rate': 4.600000000000001e-05, 'epoch': 0.01}
    {'loss': 3.4748, 'grad_norm': 5.447048664093018, 'learning_rate': 4.5833333333333334e-05, 'epoch': 0.01}
    {'loss': 3.6016, 'grad_norm': 5.365890979766846, 'learning_rate': 4.566666666666667e-05, 'epoch': 0.01}
    {'loss': 3.5459, 'grad_norm': 5.403989791870117, 'learning_rate': 4.55e-05, 'epoch': 0.01}
    {'loss': 3.6133, 'grad_norm': 4.553607940673828, 'learning_rate': 4.5333333333333335e-05, 'epoch': 0.01}
    {'loss': 3.6285, 'grad_norm': 4.794341087341309, 'learning_rate': 4.516666666666667e-05, 'epoch': 0.01}
    {'loss': 3.5379, 'grad_norm': 5.866735458374023, 'learning_rate': 4.5e-05, 'epoch': 0.01}
    {'loss': 3.4641, 'grad_norm': 5.3249006271362305, 'learning_rate': 4.483333333333333e-05, 'epoch': 0.01}
    {'loss': 3.6068, 'grad_norm': 5.728115558624268, 'learning_rate': 4.466666666666667e-05, 'epoch': 0.01}
    {'loss': 3.4145, 'grad_norm': 5.30040979385376, 'learning_rate': 4.4500000000000004e-05, 'epoch': 0.01}
    {'loss': 3.491, 'grad_norm': 5.383587837219238, 'learning_rate': 4.433333333333334e-05, 'epoch': 0.01}
    {'loss': 3.5193, 'grad_norm': 5.539207458496094, 'learning_rate': 4.4166666666666665e-05, 'epoch': 0.01}
    {'loss': 3.5744, 'grad_norm': 5.227112293243408, 'learning_rate': 4.4000000000000006e-05, 'epoch': 0.01}
    {'loss': 3.36, 'grad_norm': 4.8920392990112305, 'learning_rate': 4.383333333333334e-05, 'epoch': 0.01}
    {'loss': 3.5307, 'grad_norm': 5.161008834838867, 'learning_rate': 4.3666666666666666e-05, 'epoch': 0.01}
    {'loss': 3.5232, 'grad_norm': 5.220074653625488, 'learning_rate': 4.35e-05, 'epoch': 0.01}
    {'loss': 3.4721, 'grad_norm': 5.651122570037842, 'learning_rate': 4.3333333333333334e-05, 'epoch': 0.01}
    {'loss': 3.6852, 'grad_norm': 5.477510929107666, 'learning_rate': 4.316666666666667e-05, 'epoch': 0.01}
    {'loss': 3.4971, 'grad_norm': 5.05918025970459, 'learning_rate': 4.3e-05, 'epoch': 0.01}
    {'loss': 3.6232, 'grad_norm': 5.685992240905762, 'learning_rate': 4.2833333333333335e-05, 'epoch': 0.02}
    {'loss': 3.4166, 'grad_norm': 6.5853986740112305, 'learning_rate': 4.266666666666667e-05, 'epoch': 0.02}
    {'loss': 3.4145, 'grad_norm': 6.146334171295166, 'learning_rate': 4.25e-05, 'epoch': 0.02}
    {'loss': 3.4268, 'grad_norm': 5.708441257476807, 'learning_rate': 4.233333333333334e-05, 'epoch': 0.02}
    {'loss': 3.5307, 'grad_norm': 5.700758457183838, 'learning_rate': 4.216666666666667e-05, 'epoch': 0.02}
    {'loss': 3.4457, 'grad_norm': 7.114407539367676, 'learning_rate': 4.2e-05, 'epoch': 0.02}
    {'loss': 3.4578, 'grad_norm': 5.81282901763916, 'learning_rate': 4.183333333333334e-05, 'epoch': 0.02}
    {'loss': 3.5602, 'grad_norm': 5.977260112762451, 'learning_rate': 4.166666666666667e-05, 'epoch': 0.02}
     17%|██████▋                                 | 500/3000 [02:46<15:45,  2.64it/s]***** Running Evaluation *****
      Num examples = 50
      Batch size = 16
    
      0%|                                                     | 0/4 [00:00<?, ?it/s][A
     50%|██████████████████████▌                      | 2/4 [00:03<00:03,  1.54s/it][A
     75%|█████████████████████████████████▊           | 3/4 [00:09<00:03,  3.52s/it][A
    100%|█████████████████████████████████████████████| 4/4 [00:26<00:00,  8.69s/it][ABuilding prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 0.703 seconds.
    Prefix dict has been built successfully.
                                                                                    
    [A{'eval_rouge-1': 32.17761, 'eval_rouge-2': 7.077131999999999, 'eval_rouge-l': 24.579258, 'eval_bleu-4': 0.03384934054817397, 'eval_runtime': 45.6431, 'eval_samples_per_second': 1.095, 'eval_steps_per_second': 0.088, 'epoch': 0.02}
     17%|██████▋                                 | 500/3000 [03:32<15:45,  2.64it/s]
    100%|█████████████████████████████████████████████| 4/4 [00:27<00:00,  8.69s/it][A
    {'loss': 3.3219, 'grad_norm': 5.781159400939941, 'learning_rate': 4.15e-05, 'epoch': 0.02}
    {'loss': 3.5443, 'grad_norm': 6.7186784744262695, 'learning_rate': 4.133333333333333e-05, 'epoch': 0.02}
    {'loss': 3.5816, 'grad_norm': 5.987128257751465, 'learning_rate': 4.116666666666667e-05, 'epoch': 0.02}
    {'loss': 3.4879, 'grad_norm': 5.433833599090576, 'learning_rate': 4.1e-05, 'epoch': 0.02}
    {'loss': 3.5201, 'grad_norm': 5.364903450012207, 'learning_rate': 4.0833333333333334e-05, 'epoch': 0.02}
    {'loss': 3.6479, 'grad_norm': 5.855133533477783, 'learning_rate': 4.066666666666667e-05, 'epoch': 0.02}
    {'loss': 3.492, 'grad_norm': 5.775731563568115, 'learning_rate': 4.05e-05, 'epoch': 0.02}
    {'loss': 3.3711, 'grad_norm': 5.632279872894287, 'learning_rate': 4.0333333333333336e-05, 'epoch': 0.02}
    {'loss': 3.4236, 'grad_norm': 6.302985668182373, 'learning_rate': 4.016666666666667e-05, 'epoch': 0.02}
    {'loss': 3.4893, 'grad_norm': 6.466476917266846, 'learning_rate': 4e-05, 'epoch': 0.02}
    {'loss': 3.4369, 'grad_norm': 6.245245456695557, 'learning_rate': 3.983333333333333e-05, 'epoch': 0.02}
    {'loss': 3.4576, 'grad_norm': 6.603334903717041, 'learning_rate': 3.966666666666667e-05, 'epoch': 0.02}
    {'loss': 3.449, 'grad_norm': 6.015175819396973, 'learning_rate': 3.9500000000000005e-05, 'epoch': 0.02}
    {'loss': 3.4551, 'grad_norm': 6.189888000488281, 'learning_rate': 3.933333333333333e-05, 'epoch': 0.02}
    {'loss': 3.5338, 'grad_norm': 5.945561408996582, 'learning_rate': 3.9166666666666665e-05, 'epoch': 0.02}
    {'loss': 3.4793, 'grad_norm': 6.283290863037109, 'learning_rate': 3.9000000000000006e-05, 'epoch': 0.02}
    {'loss': 3.5412, 'grad_norm': 6.205284595489502, 'learning_rate': 3.883333333333333e-05, 'epoch': 0.02}
    {'loss': 3.3014, 'grad_norm': 6.976888179779053, 'learning_rate': 3.866666666666667e-05, 'epoch': 0.02}
    {'loss': 3.398, 'grad_norm': 6.626142501831055, 'learning_rate': 3.85e-05, 'epoch': 0.02}
    {'loss': 3.3533, 'grad_norm': 6.24869441986084, 'learning_rate': 3.8333333333333334e-05, 'epoch': 0.02}
    {'loss': 3.4967, 'grad_norm': 7.116095542907715, 'learning_rate': 3.816666666666667e-05, 'epoch': 0.02}
    {'loss': 3.5277, 'grad_norm': 6.815064907073975, 'learning_rate': 3.8e-05, 'epoch': 0.03}
    {'loss': 3.2469, 'grad_norm': 6.986637115478516, 'learning_rate': 3.7833333333333336e-05, 'epoch': 0.03}
    {'loss': 3.5721, 'grad_norm': 5.894566059112549, 'learning_rate': 3.766666666666667e-05, 'epoch': 0.03}
    {'loss': 3.3979, 'grad_norm': 6.637237071990967, 'learning_rate': 3.7500000000000003e-05, 'epoch': 0.03}
    {'loss': 3.4787, 'grad_norm': 6.173599720001221, 'learning_rate': 3.733333333333334e-05, 'epoch': 0.03}
    {'loss': 3.6227, 'grad_norm': 6.494725227355957, 'learning_rate': 3.7166666666666664e-05, 'epoch': 0.03}
    {'loss': 3.4732, 'grad_norm': 6.256740093231201, 'learning_rate': 3.7e-05, 'epoch': 0.03}
    {'loss': 3.3229, 'grad_norm': 6.459008693695068, 'learning_rate': 3.683333333333334e-05, 'epoch': 0.03}
    {'loss': 3.5484, 'grad_norm': 6.884533405303955, 'learning_rate': 3.6666666666666666e-05, 'epoch': 0.03}
    {'loss': 3.2883, 'grad_norm': 6.651428699493408, 'learning_rate': 3.65e-05, 'epoch': 0.03}
    {'loss': 3.3572, 'grad_norm': 6.499869346618652, 'learning_rate': 3.633333333333333e-05, 'epoch': 0.03}
    {'loss': 3.4582, 'grad_norm': 7.20668888092041, 'learning_rate': 3.6166666666666674e-05, 'epoch': 0.03}
    {'loss': 3.41, 'grad_norm': 6.450556755065918, 'learning_rate': 3.6e-05, 'epoch': 0.03}
    {'loss': 3.5041, 'grad_norm': 6.240203380584717, 'learning_rate': 3.5833333333333335e-05, 'epoch': 0.03}
    {'loss': 3.5322, 'grad_norm': 6.314394950866699, 'learning_rate': 3.566666666666667e-05, 'epoch': 0.03}
    {'loss': 3.2941, 'grad_norm': 7.199338912963867, 'learning_rate': 3.55e-05, 'epoch': 0.03}
    {'loss': 3.4928, 'grad_norm': 6.6620659828186035, 'learning_rate': 3.5333333333333336e-05, 'epoch': 0.03}
    {'loss': 3.452, 'grad_norm': 7.468827724456787, 'learning_rate': 3.516666666666667e-05, 'epoch': 0.03}
    {'loss': 3.2631, 'grad_norm': 8.116830825805664, 'learning_rate': 3.5e-05, 'epoch': 0.03}
    {'loss': 3.4557, 'grad_norm': 7.787027835845947, 'learning_rate': 3.483333333333334e-05, 'epoch': 0.03}
    {'loss': 3.4182, 'grad_norm': 7.112286567687988, 'learning_rate': 3.466666666666667e-05, 'epoch': 0.03}
    {'loss': 3.4584, 'grad_norm': 7.579158782958984, 'learning_rate': 3.45e-05, 'epoch': 0.03}
    {'loss': 3.5689, 'grad_norm': 7.320714950561523, 'learning_rate': 3.433333333333333e-05, 'epoch': 0.03}
    {'loss': 3.3625, 'grad_norm': 6.560305118560791, 'learning_rate': 3.4166666666666666e-05, 'epoch': 0.03}
    {'loss': 3.4361, 'grad_norm': 7.990951061248779, 'learning_rate': 3.4000000000000007e-05, 'epoch': 0.03}
    {'loss': 3.5363, 'grad_norm': 6.019072532653809, 'learning_rate': 3.3833333333333334e-05, 'epoch': 0.03}
    {'loss': 3.3223, 'grad_norm': 7.076486587524414, 'learning_rate': 3.366666666666667e-05, 'epoch': 0.03}
    {'loss': 3.4627, 'grad_norm': 7.192687034606934, 'learning_rate': 3.35e-05, 'epoch': 0.03}
    {'loss': 3.3916, 'grad_norm': 8.055869102478027, 'learning_rate': 3.3333333333333335e-05, 'epoch': 0.03}
     33%|█████████████                          | 1000/3000 [06:19<12:02,  2.77it/s]***** Running Evaluation *****
      Num examples = 50
      Batch size = 16
    
      0%|                                                     | 0/4 [00:00<?, ?it/s][A
     50%|██████████████████████▌                      | 2/4 [00:04<00:04,  2.31s/it][A
     75%|█████████████████████████████████▊           | 3/4 [00:07<00:02,  2.50s/it][A
                                                                                    [A
    [A{'eval_rouge-1': 31.829696000000002, 'eval_rouge-2': 6.706767999999999, 'eval_rouge-l': 24.602586000000002, 'eval_bleu-4': 0.03128591446385167, 'eval_runtime': 27.5903, 'eval_samples_per_second': 1.812, 'eval_steps_per_second': 0.145, 'epoch': 0.03}
     33%|█████████████                          | 1000/3000 [06:47<12:02,  2.77it/s]
    100%|█████████████████████████████████████████████| 4/4 [00:09<00:00,  2.22s/it][A
    {'loss': 3.449, 'grad_norm': 6.993073463439941, 'learning_rate': 3.316666666666667e-05, 'epoch': 0.04}
    {'loss': 3.452, 'grad_norm': 7.34697961807251, 'learning_rate': 3.3e-05, 'epoch': 0.04}
    {'loss': 3.649, 'grad_norm': 8.095622062683105, 'learning_rate': 3.283333333333333e-05, 'epoch': 0.04}
    {'loss': 3.4059, 'grad_norm': 6.550130367279053, 'learning_rate': 3.266666666666667e-05, 'epoch': 0.04}
    {'loss': 3.3877, 'grad_norm': 8.755424499511719, 'learning_rate': 3.2500000000000004e-05, 'epoch': 0.04}
    {'loss': 3.3582, 'grad_norm': 7.9052324295043945, 'learning_rate': 3.233333333333333e-05, 'epoch': 0.04}
    {'loss': 3.3902, 'grad_norm': 7.107578754425049, 'learning_rate': 3.2166666666666665e-05, 'epoch': 0.04}
    {'loss': 3.4625, 'grad_norm': 7.253876209259033, 'learning_rate': 3.2000000000000005e-05, 'epoch': 0.04}
    {'loss': 3.5266, 'grad_norm': 7.130514144897461, 'learning_rate': 3.183333333333334e-05, 'epoch': 0.04}
    {'loss': 3.4652, 'grad_norm': 6.723737716674805, 'learning_rate': 3.1666666666666666e-05, 'epoch': 0.04}
    {'loss': 3.3414, 'grad_norm': 6.968167304992676, 'learning_rate': 3.15e-05, 'epoch': 0.04}
    {'loss': 3.5258, 'grad_norm': 7.860029220581055, 'learning_rate': 3.1333333333333334e-05, 'epoch': 0.04}
    {'loss': 3.4354, 'grad_norm': 7.343764781951904, 'learning_rate': 3.116666666666667e-05, 'epoch': 0.04}
    {'loss': 3.358, 'grad_norm': 7.756207466125488, 'learning_rate': 3.1e-05, 'epoch': 0.04}
    {'loss': 3.318, 'grad_norm': 7.5618720054626465, 'learning_rate': 3.0833333333333335e-05, 'epoch': 0.04}
    {'loss': 3.3611, 'grad_norm': 7.267038822174072, 'learning_rate': 3.066666666666667e-05, 'epoch': 0.04}
    {'loss': 3.4475, 'grad_norm': 6.806843280792236, 'learning_rate': 3.05e-05, 'epoch': 0.04}
    {'loss': 3.4723, 'grad_norm': 6.421252250671387, 'learning_rate': 3.0333333333333337e-05, 'epoch': 0.04}
    {'loss': 3.3596, 'grad_norm': 6.7414631843566895, 'learning_rate': 3.016666666666667e-05, 'epoch': 0.04}
    {'loss': 3.4076, 'grad_norm': 6.416792392730713, 'learning_rate': 3e-05, 'epoch': 0.04}
    {'loss': 3.2473, 'grad_norm': 6.678061008453369, 'learning_rate': 2.9833333333333335e-05, 'epoch': 0.04}
    {'loss': 3.3428, 'grad_norm': 7.449488162994385, 'learning_rate': 2.9666666666666672e-05, 'epoch': 0.04}
    {'loss': 3.3775, 'grad_norm': 7.416559219360352, 'learning_rate': 2.95e-05, 'epoch': 0.04}
    {'loss': 3.3764, 'grad_norm': 8.8865327835083, 'learning_rate': 2.9333333333333336e-05, 'epoch': 0.04}
    {'loss': 3.4459, 'grad_norm': 6.822361946105957, 'learning_rate': 2.916666666666667e-05, 'epoch': 0.04}
    {'loss': 3.2793, 'grad_norm': 7.723625183105469, 'learning_rate': 2.9e-05, 'epoch': 0.04}
    {'loss': 3.4549, 'grad_norm': 7.177978515625, 'learning_rate': 2.8833333333333334e-05, 'epoch': 0.04}
    {'loss': 3.3346, 'grad_norm': 7.249715805053711, 'learning_rate': 2.8666666666666668e-05, 'epoch': 0.04}
    {'loss': 3.3934, 'grad_norm': 6.952620983123779, 'learning_rate': 2.8499999999999998e-05, 'epoch': 0.05}
    {'loss': 3.4795, 'grad_norm': 7.289932727813721, 'learning_rate': 2.8333333333333335e-05, 'epoch': 0.05}
    {'loss': 3.4633, 'grad_norm': 7.023521900177002, 'learning_rate': 2.816666666666667e-05, 'epoch': 0.05}
    {'loss': 3.4553, 'grad_norm': 6.813533306121826, 'learning_rate': 2.8000000000000003e-05, 'epoch': 0.05}
    {'loss': 3.4023, 'grad_norm': 10.250730514526367, 'learning_rate': 2.7833333333333333e-05, 'epoch': 0.05}
    {'loss': 3.3021, 'grad_norm': 7.482886791229248, 'learning_rate': 2.7666666666666667e-05, 'epoch': 0.05}
    {'loss': 3.3461, 'grad_norm': 7.424092769622803, 'learning_rate': 2.7500000000000004e-05, 'epoch': 0.05}
    {'loss': 3.3027, 'grad_norm': 8.092047691345215, 'learning_rate': 2.733333333333333e-05, 'epoch': 0.05}
    {'loss': 3.5166, 'grad_norm': 7.379220962524414, 'learning_rate': 2.716666666666667e-05, 'epoch': 0.05}
    {'loss': 3.3885, 'grad_norm': 7.1740312576293945, 'learning_rate': 2.7000000000000002e-05, 'epoch': 0.05}
    {'loss': 3.3533, 'grad_norm': 7.165674209594727, 'learning_rate': 2.6833333333333333e-05, 'epoch': 0.05}
    {'loss': 3.4154, 'grad_norm': 6.633505821228027, 'learning_rate': 2.6666666666666667e-05, 'epoch': 0.05}
    {'loss': 3.3449, 'grad_norm': 7.498257637023926, 'learning_rate': 2.6500000000000004e-05, 'epoch': 0.05}
    {'loss': 3.2662, 'grad_norm': 7.6140828132629395, 'learning_rate': 2.633333333333333e-05, 'epoch': 0.05}
    {'loss': 3.3785, 'grad_norm': 7.697851657867432, 'learning_rate': 2.6166666666666668e-05, 'epoch': 0.05}
    {'loss': 3.3557, 'grad_norm': 7.371003150939941, 'learning_rate': 2.6000000000000002e-05, 'epoch': 0.05}
    {'loss': 3.2658, 'grad_norm': 7.067782878875732, 'learning_rate': 2.5833333333333336e-05, 'epoch': 0.05}
    {'loss': 3.3914, 'grad_norm': 7.24818754196167, 'learning_rate': 2.5666666666666666e-05, 'epoch': 0.05}
    {'loss': 3.435, 'grad_norm': 9.331357955932617, 'learning_rate': 2.5500000000000003e-05, 'epoch': 0.05}
    {'loss': 3.2973, 'grad_norm': 6.749776363372803, 'learning_rate': 2.5333333333333337e-05, 'epoch': 0.05}
    {'loss': 3.4361, 'grad_norm': 7.432412147521973, 'learning_rate': 2.5166666666666667e-05, 'epoch': 0.05}
    {'loss': 3.4547, 'grad_norm': 6.918740272521973, 'learning_rate': 2.5e-05, 'epoch': 0.05}
     50%|███████████████████▌                   | 1500/3000 [09:34<07:31,  3.32it/s]***** Running Evaluation *****
      Num examples = 50
      Batch size = 16
    
      0%|                                                     | 0/4 [00:00<?, ?it/s][A
     50%|██████████████████████▌                      | 2/4 [00:03<00:03,  1.69s/it][A
     75%|█████████████████████████████████▊           | 3/4 [00:06<00:02,  2.22s/it][A
                                                                                    [A
    [A{'eval_rouge-1': 32.157004, 'eval_rouge-2': 6.817944000000001, 'eval_rouge-l': 25.710727999999996, 'eval_bleu-4': 0.03216092782545496, 'eval_runtime': 12.6369, 'eval_samples_per_second': 3.957, 'eval_steps_per_second': 0.317, 'epoch': 0.05}
     50%|███████████████████▌                   | 1500/3000 [09:46<07:31,  3.32it/s]
    100%|█████████████████████████████████████████████| 4/4 [00:08<00:00,  2.31s/it][A
    {'loss': 3.3398, 'grad_norm': 6.808885097503662, 'learning_rate': 2.4833333333333335e-05, 'epoch': 0.05}
    {'loss': 3.3883, 'grad_norm': 8.204680442810059, 'learning_rate': 2.466666666666667e-05, 'epoch': 0.05}
    {'loss': 3.4418, 'grad_norm': 8.245332717895508, 'learning_rate': 2.45e-05, 'epoch': 0.05}
    {'loss': 3.4062, 'grad_norm': 7.062271595001221, 'learning_rate': 2.4333333333333336e-05, 'epoch': 0.05}
    {'loss': 3.493, 'grad_norm': 7.276569843292236, 'learning_rate': 2.4166666666666667e-05, 'epoch': 0.05}
    {'loss': 3.4061, 'grad_norm': 8.414140701293945, 'learning_rate': 2.4e-05, 'epoch': 0.05}
    {'loss': 3.4645, 'grad_norm': 7.905790328979492, 'learning_rate': 2.3833333333333334e-05, 'epoch': 0.05}
    {'loss': 3.4293, 'grad_norm': 7.57233190536499, 'learning_rate': 2.3666666666666668e-05, 'epoch': 0.06}
    {'loss': 3.5066, 'grad_norm': 9.469871520996094, 'learning_rate': 2.35e-05, 'epoch': 0.06}
    {'loss': 3.3914, 'grad_norm': 6.963181972503662, 'learning_rate': 2.3333333333333336e-05, 'epoch': 0.06}
    {'loss': 3.3687, 'grad_norm': 8.012818336486816, 'learning_rate': 2.3166666666666666e-05, 'epoch': 0.06}
    {'loss': 3.3639, 'grad_norm': 8.616475105285645, 'learning_rate': 2.3000000000000003e-05, 'epoch': 0.06}
    {'loss': 3.4703, 'grad_norm': 7.470798492431641, 'learning_rate': 2.2833333333333334e-05, 'epoch': 0.06}
    {'loss': 3.3197, 'grad_norm': 8.026731491088867, 'learning_rate': 2.2666666666666668e-05, 'epoch': 0.06}
    {'loss': 3.3693, 'grad_norm': 7.616352558135986, 'learning_rate': 2.25e-05, 'epoch': 0.06}
    {'loss': 3.2992, 'grad_norm': 7.114800930023193, 'learning_rate': 2.2333333333333335e-05, 'epoch': 0.06}
    {'loss': 3.4791, 'grad_norm': 8.939187049865723, 'learning_rate': 2.216666666666667e-05, 'epoch': 0.06}
    {'loss': 3.3701, 'grad_norm': 7.229987144470215, 'learning_rate': 2.2000000000000003e-05, 'epoch': 0.06}
    {'loss': 3.3725, 'grad_norm': 7.43455696105957, 'learning_rate': 2.1833333333333333e-05, 'epoch': 0.06}
    {'loss': 3.5133, 'grad_norm': 7.135754585266113, 'learning_rate': 2.1666666666666667e-05, 'epoch': 0.06}
    {'loss': 3.4623, 'grad_norm': 7.312079906463623, 'learning_rate': 2.15e-05, 'epoch': 0.06}
    {'loss': 3.5047, 'grad_norm': 7.399215221405029, 'learning_rate': 2.1333333333333335e-05, 'epoch': 0.06}
    {'loss': 3.4035, 'grad_norm': 7.4649152755737305, 'learning_rate': 2.116666666666667e-05, 'epoch': 0.06}
    {'loss': 3.402, 'grad_norm': 7.635732650756836, 'learning_rate': 2.1e-05, 'epoch': 0.06}
    {'loss': 3.4635, 'grad_norm': 7.484546184539795, 'learning_rate': 2.0833333333333336e-05, 'epoch': 0.06}
    {'loss': 3.4439, 'grad_norm': 7.9170403480529785, 'learning_rate': 2.0666666666666666e-05, 'epoch': 0.06}
    {'loss': 3.3592, 'grad_norm': 8.43758773803711, 'learning_rate': 2.05e-05, 'epoch': 0.06}
    {'loss': 3.3473, 'grad_norm': 8.31228256225586, 'learning_rate': 2.0333333333333334e-05, 'epoch': 0.06}
    {'loss': 3.3908, 'grad_norm': 8.26220703125, 'learning_rate': 2.0166666666666668e-05, 'epoch': 0.06}
    {'loss': 3.34, 'grad_norm': 7.91382360458374, 'learning_rate': 2e-05, 'epoch': 0.06}
    {'loss': 3.3785, 'grad_norm': 9.013530731201172, 'learning_rate': 1.9833333333333335e-05, 'epoch': 0.06}
    {'loss': 3.3432, 'grad_norm': 7.767002105712891, 'learning_rate': 1.9666666666666666e-05, 'epoch': 0.06}
    {'loss': 3.5836, 'grad_norm': 7.97884464263916, 'learning_rate': 1.9500000000000003e-05, 'epoch': 0.06}
    {'loss': 3.3471, 'grad_norm': 8.59230899810791, 'learning_rate': 1.9333333333333333e-05, 'epoch': 0.06}
    {'loss': 3.4852, 'grad_norm': 8.9595308303833, 'learning_rate': 1.9166666666666667e-05, 'epoch': 0.06}
    {'loss': 3.3811, 'grad_norm': 7.3524699211120605, 'learning_rate': 1.9e-05, 'epoch': 0.06}
    {'loss': 3.3145, 'grad_norm': 8.164934158325195, 'learning_rate': 1.8833333333333335e-05, 'epoch': 0.07}
    {'loss': 3.3039, 'grad_norm': 7.434097766876221, 'learning_rate': 1.866666666666667e-05, 'epoch': 0.07}
    {'loss': 3.4008, 'grad_norm': 7.333683490753174, 'learning_rate': 1.85e-05, 'epoch': 0.07}
    {'loss': 3.367, 'grad_norm': 7.950295925140381, 'learning_rate': 1.8333333333333333e-05, 'epoch': 0.07}
    {'loss': 3.3881, 'grad_norm': 8.113795280456543, 'learning_rate': 1.8166666666666667e-05, 'epoch': 0.07}
    {'loss': 3.485, 'grad_norm': 7.562678813934326, 'learning_rate': 1.8e-05, 'epoch': 0.07}
    {'loss': 3.2814, 'grad_norm': 7.7911152839660645, 'learning_rate': 1.7833333333333334e-05, 'epoch': 0.07}
    {'loss': 3.5012, 'grad_norm': 7.548689365386963, 'learning_rate': 1.7666666666666668e-05, 'epoch': 0.07}
    {'loss': 3.3568, 'grad_norm': 6.933331489562988, 'learning_rate': 1.75e-05, 'epoch': 0.07}
    {'loss': 3.283, 'grad_norm': 8.710576057434082, 'learning_rate': 1.7333333333333336e-05, 'epoch': 0.07}
    {'loss': 3.3713, 'grad_norm': 7.758098602294922, 'learning_rate': 1.7166666666666666e-05, 'epoch': 0.07}
    {'loss': 3.2377, 'grad_norm': 7.713993549346924, 'learning_rate': 1.7000000000000003e-05, 'epoch': 0.07}
    {'loss': 3.4193, 'grad_norm': 7.195184230804443, 'learning_rate': 1.6833333333333334e-05, 'epoch': 0.07}
    {'loss': 3.4684, 'grad_norm': 8.01244831085205, 'learning_rate': 1.6666666666666667e-05, 'epoch': 0.07}
     67%|██████████████████████████             | 2000/3000 [12:33<05:27,  3.05it/s]***** Running Evaluation *****
      Num examples = 50
      Batch size = 16
    
      0%|                                                     | 0/4 [00:00<?, ?it/s][A
     50%|██████████████████████▌                      | 2/4 [00:17<00:17,  8.67s/it][A
     75%|█████████████████████████████████▊           | 3/4 [00:34<00:12, 12.24s/it][A
                                                                                    [A
    [A{'eval_rouge-1': 30.446856000000004, 'eval_rouge-2': 6.523476, 'eval_rouge-l': 22.410238, 'eval_bleu-4': 0.02954999477763607, 'eval_runtime': 70.2014, 'eval_samples_per_second': 0.712, 'eval_steps_per_second': 0.057, 'epoch': 0.07}
     67%|██████████████████████████             | 2000/3000 [13:44<05:27,  3.05it/s]
    100%|█████████████████████████████████████████████| 4/4 [00:52<00:00, 14.13s/it][A
                                                                                    [ASaving model checkpoint to ./output/checkpoint-2000
    /root/miniconda3/lib/python3.10/site-packages/peft/utils/save_and_load.py:154: UserWarning: Could not find a config file in /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b - will assume that the vocabulary was not modified.
      warnings.warn(
    /root/miniconda3/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
      warnings.warn(
    {'loss': 3.3824, 'grad_norm': 8.776557922363281, 'learning_rate': 1.65e-05, 'epoch': 0.07}
    {'loss': 3.4982, 'grad_norm': 7.443075180053711, 'learning_rate': 1.6333333333333335e-05, 'epoch': 0.07}
    {'loss': 3.5551, 'grad_norm': 9.008573532104492, 'learning_rate': 1.6166666666666665e-05, 'epoch': 0.07}
    {'loss': 3.4877, 'grad_norm': 8.442090034484863, 'learning_rate': 1.6000000000000003e-05, 'epoch': 0.07}
    {'loss': 3.3713, 'grad_norm': 8.272077560424805, 'learning_rate': 1.5833333333333333e-05, 'epoch': 0.07}
    {'loss': 3.3291, 'grad_norm': 7.87449836730957, 'learning_rate': 1.5666666666666667e-05, 'epoch': 0.07}
    {'loss': 3.4443, 'grad_norm': 8.297041893005371, 'learning_rate': 1.55e-05, 'epoch': 0.07}
    {'loss': 3.4166, 'grad_norm': 8.197674751281738, 'learning_rate': 1.5333333333333334e-05, 'epoch': 0.07}
    {'loss': 3.4396, 'grad_norm': 7.396449089050293, 'learning_rate': 1.5166666666666668e-05, 'epoch': 0.07}
    {'loss': 3.3553, 'grad_norm': 7.69448709487915, 'learning_rate': 1.5e-05, 'epoch': 0.07}
    {'loss': 3.2928, 'grad_norm': 7.604002952575684, 'learning_rate': 1.4833333333333336e-05, 'epoch': 0.07}
    {'loss': 3.585, 'grad_norm': 7.937329292297363, 'learning_rate': 1.4666666666666668e-05, 'epoch': 0.07}
    {'loss': 3.2537, 'grad_norm': 7.553071975708008, 'learning_rate': 1.45e-05, 'epoch': 0.07}
    {'loss': 3.3604, 'grad_norm': 8.35810661315918, 'learning_rate': 1.4333333333333334e-05, 'epoch': 0.07}
    {'loss': 3.3957, 'grad_norm': 7.448248863220215, 'learning_rate': 1.4166666666666668e-05, 'epoch': 0.08}
    {'loss': 3.5211, 'grad_norm': 8.03997802734375, 'learning_rate': 1.4000000000000001e-05, 'epoch': 0.08}
    {'loss': 3.3982, 'grad_norm': 6.854673385620117, 'learning_rate': 1.3833333333333334e-05, 'epoch': 0.08}
    {'loss': 3.4145, 'grad_norm': 7.920439720153809, 'learning_rate': 1.3666666666666666e-05, 'epoch': 0.08}
    {'loss': 3.3545, 'grad_norm': 7.609816074371338, 'learning_rate': 1.3500000000000001e-05, 'epoch': 0.08}
    {'loss': 3.4355, 'grad_norm': 7.644512176513672, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.08}
    {'loss': 3.4512, 'grad_norm': 6.954286098480225, 'learning_rate': 1.3166666666666665e-05, 'epoch': 0.08}
    {'loss': 3.4193, 'grad_norm': 7.823613166809082, 'learning_rate': 1.3000000000000001e-05, 'epoch': 0.08}
    {'loss': 3.4156, 'grad_norm': 8.042947769165039, 'learning_rate': 1.2833333333333333e-05, 'epoch': 0.08}
    {'loss': 3.3693, 'grad_norm': 8.20654582977295, 'learning_rate': 1.2666666666666668e-05, 'epoch': 0.08}
    {'loss': 3.2385, 'grad_norm': 8.42420482635498, 'learning_rate': 1.25e-05, 'epoch': 0.08}
    {'loss': 3.3656, 'grad_norm': 7.991057872772217, 'learning_rate': 1.2333333333333334e-05, 'epoch': 0.08}
    {'loss': 3.4273, 'grad_norm': 8.669829368591309, 'learning_rate': 1.2166666666666668e-05, 'epoch': 0.08}
    {'loss': 3.4666, 'grad_norm': 7.730335712432861, 'learning_rate': 1.2e-05, 'epoch': 0.08}
    {'loss': 3.2955, 'grad_norm': 8.409688949584961, 'learning_rate': 1.1833333333333334e-05, 'epoch': 0.08}
    {'loss': 3.358, 'grad_norm': 8.440476417541504, 'learning_rate': 1.1666666666666668e-05, 'epoch': 0.08}
    {'loss': 3.3174, 'grad_norm': 8.47851848602295, 'learning_rate': 1.1500000000000002e-05, 'epoch': 0.08}
    {'loss': 3.3273, 'grad_norm': 8.534693717956543, 'learning_rate': 1.1333333333333334e-05, 'epoch': 0.08}
    {'loss': 3.3674, 'grad_norm': 9.091329574584961, 'learning_rate': 1.1166666666666668e-05, 'epoch': 0.08}
    {'loss': 3.3656, 'grad_norm': 7.699550151824951, 'learning_rate': 1.1000000000000001e-05, 'epoch': 0.08}
    {'loss': 3.2736, 'grad_norm': 8.524870872497559, 'learning_rate': 1.0833333333333334e-05, 'epoch': 0.08}
    {'loss': 3.3752, 'grad_norm': 8.285065650939941, 'learning_rate': 1.0666666666666667e-05, 'epoch': 0.08}
    {'loss': 3.349, 'grad_norm': 7.8302459716796875, 'learning_rate': 1.05e-05, 'epoch': 0.08}
    {'loss': 3.4945, 'grad_norm': 8.535857200622559, 'learning_rate': 1.0333333333333333e-05, 'epoch': 0.08}
    {'loss': 3.2322, 'grad_norm': 8.664883613586426, 'learning_rate': 1.0166666666666667e-05, 'epoch': 0.08}
    {'loss': 3.4578, 'grad_norm': 7.750856399536133, 'learning_rate': 1e-05, 'epoch': 0.08}
    {'loss': 3.4516, 'grad_norm': 8.10501766204834, 'learning_rate': 9.833333333333333e-06, 'epoch': 0.08}
    {'loss': 3.2816, 'grad_norm': 8.05683708190918, 'learning_rate': 9.666666666666667e-06, 'epoch': 0.08}
    {'loss': 3.3736, 'grad_norm': 7.280567169189453, 'learning_rate': 9.5e-06, 'epoch': 0.08}
    {'loss': 3.3818, 'grad_norm': 8.007243156433105, 'learning_rate': 9.333333333333334e-06, 'epoch': 0.09}
    {'loss': 3.2754, 'grad_norm': 7.657828330993652, 'learning_rate': 9.166666666666666e-06, 'epoch': 0.09}
    {'loss': 3.3133, 'grad_norm': 7.7801361083984375, 'learning_rate': 9e-06, 'epoch': 0.09}
    {'loss': 3.26, 'grad_norm': 8.849067687988281, 'learning_rate': 8.833333333333334e-06, 'epoch': 0.09}
    {'loss': 3.4408, 'grad_norm': 7.51537561416626, 'learning_rate': 8.666666666666668e-06, 'epoch': 0.09}
    {'loss': 3.4723, 'grad_norm': 7.8354668617248535, 'learning_rate': 8.500000000000002e-06, 'epoch': 0.09}
    {'loss': 3.3928, 'grad_norm': 9.467879295349121, 'learning_rate': 8.333333333333334e-06, 'epoch': 0.09}
     83%|████████████████████████████████▌      | 2500/3000 [16:30<02:42,  3.07it/s]***** Running Evaluation *****
      Num examples = 50
      Batch size = 16
    
      0%|                                                     | 0/4 [00:00<?, ?it/s][A
     50%|██████████████████████▌                      | 2/4 [00:03<00:03,  1.82s/it][A
     75%|█████████████████████████████████▊           | 3/4 [00:21<00:08,  8.43s/it][A
                                                                                    [A
    [A{'eval_rouge-1': 32.296754, 'eval_rouge-2': 7.134015999999998, 'eval_rouge-l': 23.826210000000007, 'eval_bleu-4': 0.031113838053489586, 'eval_runtime': 57.4732, 'eval_samples_per_second': 0.87, 'eval_steps_per_second': 0.07, 'epoch': 0.09}
     83%|████████████████████████████████▌      | 2500/3000 [17:28<02:42,  3.07it/s]
    100%|█████████████████████████████████████████████| 4/4 [00:39<00:00, 11.92s/it][A
    {'loss': 3.3018, 'grad_norm': 8.49243450164795, 'learning_rate': 8.166666666666668e-06, 'epoch': 0.09}
    {'loss': 3.3375, 'grad_norm': 9.953892707824707, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.09}
    {'loss': 3.2449, 'grad_norm': 7.946211338043213, 'learning_rate': 7.833333333333333e-06, 'epoch': 0.09}
    {'loss': 3.402, 'grad_norm': 8.16499137878418, 'learning_rate': 7.666666666666667e-06, 'epoch': 0.09}
    {'loss': 3.3908, 'grad_norm': 7.735748767852783, 'learning_rate': 7.5e-06, 'epoch': 0.09}
    {'loss': 3.4027, 'grad_norm': 8.423108100891113, 'learning_rate': 7.333333333333334e-06, 'epoch': 0.09}
    {'loss': 3.4703, 'grad_norm': 7.9892754554748535, 'learning_rate': 7.166666666666667e-06, 'epoch': 0.09}
    {'loss': 3.4754, 'grad_norm': 8.66110610961914, 'learning_rate': 7.000000000000001e-06, 'epoch': 0.09}
    {'loss': 3.374, 'grad_norm': 8.421737670898438, 'learning_rate': 6.833333333333333e-06, 'epoch': 0.09}
    {'loss': 3.4771, 'grad_norm': 8.791631698608398, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.09}
    {'loss': 3.3588, 'grad_norm': 8.110115051269531, 'learning_rate': 6.5000000000000004e-06, 'epoch': 0.09}
    {'loss': 3.424, 'grad_norm': 7.765480041503906, 'learning_rate': 6.333333333333334e-06, 'epoch': 0.09}
    {'loss': 3.5215, 'grad_norm': 7.676518440246582, 'learning_rate': 6.166666666666667e-06, 'epoch': 0.09}
    {'loss': 3.4461, 'grad_norm': 8.662293434143066, 'learning_rate': 6e-06, 'epoch': 0.09}
    {'loss': 3.4045, 'grad_norm': 8.041947364807129, 'learning_rate': 5.833333333333334e-06, 'epoch': 0.09}
    {'loss': 3.35, 'grad_norm': 7.886188507080078, 'learning_rate': 5.666666666666667e-06, 'epoch': 0.09}
    {'loss': 3.4121, 'grad_norm': 8.588319778442383, 'learning_rate': 5.500000000000001e-06, 'epoch': 0.09}
    {'loss': 3.2682, 'grad_norm': 7.643811225891113, 'learning_rate': 5.333333333333334e-06, 'epoch': 0.09}
    {'loss': 3.4762, 'grad_norm': 8.756789207458496, 'learning_rate': 5.166666666666667e-06, 'epoch': 0.09}
    {'loss': 3.4521, 'grad_norm': 8.896966934204102, 'learning_rate': 5e-06, 'epoch': 0.09}
    {'loss': 3.4303, 'grad_norm': 8.418134689331055, 'learning_rate': 4.833333333333333e-06, 'epoch': 0.09}
    {'loss': 3.2492, 'grad_norm': 7.734707832336426, 'learning_rate': 4.666666666666667e-06, 'epoch': 0.09}
    {'loss': 3.3811, 'grad_norm': 8.035101890563965, 'learning_rate': 4.5e-06, 'epoch': 0.1}
    {'loss': 3.3842, 'grad_norm': 8.145923614501953, 'learning_rate': 4.333333333333334e-06, 'epoch': 0.1}
    {'loss': 3.4566, 'grad_norm': 8.874634742736816, 'learning_rate': 4.166666666666667e-06, 'epoch': 0.1}
    {'loss': 3.3969, 'grad_norm': 8.283893585205078, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.1}
    {'loss': 3.3459, 'grad_norm': 8.323463439941406, 'learning_rate': 3.833333333333334e-06, 'epoch': 0.1}
    {'loss': 3.2607, 'grad_norm': 8.614544868469238, 'learning_rate': 3.666666666666667e-06, 'epoch': 0.1}
    {'loss': 3.282, 'grad_norm': 8.1693115234375, 'learning_rate': 3.5000000000000004e-06, 'epoch': 0.1}
    {'loss': 3.2412, 'grad_norm': 7.729308128356934, 'learning_rate': 3.3333333333333333e-06, 'epoch': 0.1}
    {'loss': 3.4469, 'grad_norm': 7.971092224121094, 'learning_rate': 3.166666666666667e-06, 'epoch': 0.1}
    {'loss': 3.3715, 'grad_norm': 8.092921257019043, 'learning_rate': 3e-06, 'epoch': 0.1}
    {'loss': 3.3924, 'grad_norm': 7.977327823638916, 'learning_rate': 2.8333333333333335e-06, 'epoch': 0.1}
    {'loss': 3.443, 'grad_norm': 9.199762344360352, 'learning_rate': 2.666666666666667e-06, 'epoch': 0.1}
    {'loss': 3.409, 'grad_norm': 8.606124877929688, 'learning_rate': 2.5e-06, 'epoch': 0.1}
    {'loss': 3.3404, 'grad_norm': 8.420580863952637, 'learning_rate': 2.3333333333333336e-06, 'epoch': 0.1}
    {'loss': 3.3752, 'grad_norm': 8.722784996032715, 'learning_rate': 2.166666666666667e-06, 'epoch': 0.1}
    {'loss': 3.5109, 'grad_norm': 9.062599182128906, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.1}
    {'loss': 3.3029, 'grad_norm': 8.110153198242188, 'learning_rate': 1.8333333333333335e-06, 'epoch': 0.1}
    {'loss': 3.3281, 'grad_norm': 8.998739242553711, 'learning_rate': 1.6666666666666667e-06, 'epoch': 0.1}
    {'loss': 3.308, 'grad_norm': 8.037696838378906, 'learning_rate': 1.5e-06, 'epoch': 0.1}
    {'loss': 3.2516, 'grad_norm': 7.361093521118164, 'learning_rate': 1.3333333333333334e-06, 'epoch': 0.1}
    {'loss': 3.3576, 'grad_norm': 9.00409984588623, 'learning_rate': 1.1666666666666668e-06, 'epoch': 0.1}
    {'loss': 3.2564, 'grad_norm': 8.653921127319336, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.1}
    {'loss': 3.3779, 'grad_norm': 8.067493438720703, 'learning_rate': 8.333333333333333e-07, 'epoch': 0.1}
    {'loss': 3.207, 'grad_norm': 9.163408279418945, 'learning_rate': 6.666666666666667e-07, 'epoch': 0.1}
    {'loss': 3.449, 'grad_norm': 8.909379959106445, 'learning_rate': 5.000000000000001e-07, 'epoch': 0.1}
    {'loss': 3.4309, 'grad_norm': 8.986862182617188, 'learning_rate': 3.3333333333333335e-07, 'epoch': 0.1}
    {'loss': 3.4771, 'grad_norm': 8.072227478027344, 'learning_rate': 1.6666666666666668e-07, 'epoch': 0.1}
    {'loss': 3.3686, 'grad_norm': 7.826257705688477, 'learning_rate': 0.0, 'epoch': 0.1}
    100%|███████████████████████████████████████| 3000/3000 [20:15<00:00,  3.16it/s]***** Running Evaluation *****
      Num examples = 50
      Batch size = 16
    
      0%|                                                     | 0/4 [00:00<?, ?it/s][A
     50%|██████████████████████▌                      | 2/4 [00:04<00:04,  2.22s/it][A
     75%|█████████████████████████████████▊           | 3/4 [00:22<00:08,  8.75s/it][A
                                                                                    [A
    [A{'eval_rouge-1': 31.518654, 'eval_rouge-2': 6.8847700000000005, 'eval_rouge-l': 23.27281, 'eval_bleu-4': 0.031192578614046812, 'eval_runtime': 58.8645, 'eval_samples_per_second': 0.849, 'eval_steps_per_second': 0.068, 'epoch': 0.1}
    100%|███████████████████████████████████████| 3000/3000 [21:13<00:00,  3.16it/s]
    100%|█████████████████████████████████████████████| 4/4 [00:40<00:00, 12.15s/it][A
                                                                                    [A
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    
    {'train_runtime': 1274.0209, 'train_samples_per_second': 9.419, 'train_steps_per_second': 2.355, 'train_loss': 3.4456588541666666, 'epoch': 0.1}
    100%|███████████████████████████████████████| 3000/3000 [21:14<00:00,  2.35it/s]
    ***** Running Prediction *****
      Num examples = 1070
      Batch size = 16
     21%|████████▉                                  | 14/67 [02:22<09:17, 10.53s/it]


```python
# 临时关闭了电脑，中断了输出，但是训练完成了
```


```python
ll output
```

    total 4
    drwxr-xr-x 2 root 4096 Apr  3 08:06 [0m[01;34mcheckpoint-2000[0m/
    drwxr-xr-x 3 root   73 Apr  3 07:52 [01;34mruns[0m/



```python
ll output/checkpoint-2000
```

    total 22984
    -rw-r--r-- 1 root     5121 Apr  3 08:06 README.md
    -rw-r--r-- 1 root      666 Apr  3 08:06 adapter_config.json
    -rw-r--r-- 1 root  7807744 Apr  3 08:06 adapter_model.safetensors
    -rw-r--r-- 1 root 15644922 Apr  3 08:06 optimizer.pt
    -rw-r--r-- 1 root    14244 Apr  3 08:06 rng_state.pth
    -rw-r--r-- 1 root     1064 Apr  3 08:06 scheduler.pt
    -rw-r--r-- 1 root    32899 Apr  3 08:06 trainer_state.json
    -rw-r--r-- 1 root     6584 Apr  3 08:06 training_args.bin


# 备份Checkpoint


```python
!zip -r output-3000.zip output/
```

    updating: output/ (stored 0%)
    updating: output/runs/ (stored 0%)
    updating: output/runs/Apr03_07-52-07_autodl-container-4c754cbcba-ce8e9adc/ (stored 0%)
    updating: output/runs/Apr03_07-52-07_autodl-container-4c754cbcba-ce8e9adc/events.out.tfevents.1712101941.autodl-container-4c754cbcba-ce8e9adc.11916.0 (deflated 71%)
    updating: output/checkpoint-2000/ (stored 0%)
    updating: output/checkpoint-2000/README.md (deflated 66%)
    updating: output/checkpoint-2000/adapter_model.safetensors (deflated 7%)
    updating: output/checkpoint-2000/adapter_config.json (deflated 50%)
    updating: output/checkpoint-2000/training_args.bin (deflated 52%)
    updating: output/checkpoint-2000/optimizer.pt (deflated 8%)
    updating: output/checkpoint-2000/scheduler.pt (deflated 55%)
    updating: output/checkpoint-2000/rng_state.pth (deflated 25%)
    updating: output/checkpoint-2000/trainer_state.json (deflated 84%)


# 使用微调后的模型

## 在 inference_hf.py 中验证微调后的模型

您可以在 `finetune_demo/inference_hf.py` 中使用我们的微调后的模型，仅需要一行代码就能简单的进行测试。


```python
cat output/checkpoint-2000/adapter_config.json
```

    {
      "alpha_pattern": {},
      "auto_mapping": null,
      "base_model_name_or_path": "/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b",
      "bias": "none",
      "fan_in_fan_out": false,
      "inference_mode": true,
      "init_lora_weights": true,
      "layer_replication": null,
      "layers_pattern": null,
      "layers_to_transform": null,
      "loftq_config": {},
      "lora_alpha": 64,
      "lora_dropout": 0.1,
      "megatron_config": null,
      "megatron_core": "megatron.core",
      "modules_to_save": null,
      "peft_type": "LORA",
      "r": 16,
      "rank_pattern": {},
      "revision": null,
      "target_modules": [
        "query_key_value"
      ],
      "task_type": "CAUSAL_LM",
      "use_dora": false,
      "use_rslora": false
    }


```python
!python inference_hf.py /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b --prompt 类型#裙*裙长#半身裙
```

    Loading checkpoint shards:   0%|                          | 0/7 [00:00<?, ?it/s]/root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    Loading checkpoint shards: 100%|██████████████████| 7/7 [00:09<00:00,  1.42s/it]
    #裙长# 半身裙是一种长度及下摆位置介于膝盖上方和脚踝之间的裙子。这种裙子可以展现女性的优美曲线和优雅气质，适合各种场合穿着。在选择半身裙时，可以根据自己的身高、体型和喜好挑选不同长度、颜色和材质的裙子。



```python
!python inference_hf.py output/checkpoint-2000/ --prompt 类型#裙*裙长#半身裙
```

    Loading checkpoint shards:   0%|                          | 0/7 [00:00<?, ?it/s]/root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    Loading checkpoint shards: 100%|██████████████████| 7/7 [00:04<00:00,  1.54it/s]
    这款半身裙采用优质的面料制作，具有很好的透气性和吸汗性，穿着舒适。裙身采用松紧腰带设计，能够随意调节腰围大小，方便穿搭。裙摆采用不规则的剪裁设计，具有很好的层次感，穿着显高显腿。



```python
!python inference_hf.py output/checkpoint-2000/ --prompt 类型#裙*裙长#半身裙
```

    Loading checkpoint shards:   0%|                          | 0/7 [00:00<?, ?it/s]/root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    Loading checkpoint shards: 100%|██████████████████| 7/7 [00:04<00:00,  1.67it/s]
    半身裙作为日常穿搭的必备单品，经典百搭。这条半身裙采用纯棉面料，柔软舒适，舒适度较高。裙身采用纯色设计，简约大气，搭配起来非常简单。裙摆处采用不规则的荷叶边设计，增添层次感，提升气质。



```python

```


```python

```


```python

```


```python

```
