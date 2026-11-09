# Copyright 2024 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

libMyKernels.so: myKernels.o
	g++ -shared myKernels.o -o libMyKernels.so

myKernels.o: myKernels.cpp
	g++ -c -fPIC myKernels.cpp -I../../../../../src -std=c++17 -o myKernels.o
    
clean:
	rm -rf myKernels.o libMyKernels.so