package org.trifort.rootbeer.sort;

import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.CacheConfig;
import java.util.List;
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;

public class GPUSort {

  private int[] newArray(int size){
    int[] ret = new int[size];

    for(int i = 0; i < size; ++i){
      ret[i] = i;
    }
    return ret;
  }

  public void checkSorted(int[] array, int outerIndex){
    for(int index = 0; index < array.length; ++index){
      if(array[index] != index){
        for(int index2 = 0; index2 < array.length; ++index2){
          System.out.println("array["+index2+"]: "+array[index2]);
        }
        throw new RuntimeException("not sorted: "+outerIndex);
      }
    }
  }

  public void fisherYates(int[] array)
  {
    Random random = new Random();
    for (int i = array.length - 1; i > 0; i--){
    	/* Returns a pseudorandom, uniformly distributed int value 
    	   between 0 (inclusive) and the specified value (exclusive), 
    	   drawn from this random number generator's sequence. */
      int index = random.nextInt(i + 1); // one random index
      int a = array[index]; // content of the field
      array[index] = array[i]; // set content of the field to content of the current i field
      array[i] = a; // set content of the current i field to the old content of the field
    }
  }

  // Titan> http://www.anandtech.com/show/6760/nvidias-geforce-gtx-titan-part-1/4
  // 2888 Cores, 6143 MB Dedicated Memory
  // 32 threads / warp | 64 warps / SM
  // 2048 threads / SM
  // 255kb register per thread
  // 64 FP64 CUDA cores -> 2 FP64 SM
  // 14 SM 
  
  public void sort(){

    //should have 192 threads per SM
	int numberOfRuns = 1;  
	// inner array size
    int size = 2048;
    int sizeBy2 = size / 2;
//    int numMultiProcessors = 14;
//    int blocksPerMultiProcessor = 512;
    int numMultiProcessors = 2;
    int blocksPerMultiProcessor = 256;

    // set size of the outer array to be
    int outerCount = numMultiProcessors*blocksPerMultiProcessor;
    int[][] array = new int[outerCount][];
    // create array as wide as the number of SM * the blocks per SM
    // create one size wide inner array per block(thread group)
    for(int i = 0; i < outerCount; ++i){
      array[i] = newArray(size);
    }
    
    Rootbeer rootbeer = new Rootbeer();
    List<GpuDevice> devices = rootbeer.getDevices();
    GpuDevice device0 = devices.get(0);
    //create a context with 4212880 bytes objectMemory.
    //you can leave the 4212880 missing at first to
    //use all available GPU memory. after you run you
    //can call context0.getRequiredMemory() to see
    //what value to enter here
    Context context0 = device0.createContext(); // 4212880
  //use more die area for shared memory instead of
    //cache. the shared memory is a software defined
    //cache that, if programmed properly, can perform
    //better than the hardware cache
    //see (CUDA Occupancy calculator)[http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls]
    context0.setCacheConfig(CacheConfig.PREFER_SHARED);
    //wire thread config for throughput mode. after
    //calling buildState, the book-keeping information
    //will be cached in the JNI driver
    context0.setThreadConfig(sizeBy2, outerCount, outerCount * sizeBy2);
    //configure to use kernel templates. rather than
    //using kernel lists where each thread has a Kernel
    //object, there is only one kernel object (less memory copies)
    //when using kernel templates you need to differetiate
    //your data using thread/block indexes
    context0.setKernel(new GPUSortKernel(array));
    //cache the state and get ready for throughput mode
    context0.buildState();
    
    List<Double> ratioList = new ArrayList<Double>(1);
    int runs = 0;

    // limit the run
    while(runs < numberOfRuns){
      runs += 1;
      System.out.println("Run "+runs+" start");

//  while(true){
      //randomize the array to be sorted


      for(int i = 0; i < outerCount; ++i){
        fisherYates(array[i]);
      }
//      System.out.println(Arrays.deepToString(array));
      long gpuStart = System.currentTimeMillis();
      //run the cached throughput mode state.
      //the data now reachable from the only
      //GPUSortKernel is serialized to the GPU
      context0.run();

      // stats and stat output
      long gpuStop = System.currentTimeMillis();
      long gpuTime = gpuStop - gpuStart;
      StatsRow row0 = context0.getStats();
      System.out.println("serialization_time: "+row0.getSerializationTime());
      System.out.println("driver_memcopy_to_device_time: "+row0.getDriverMemcopyToDeviceTime());
      System.out.println("driver_execution_time: "+row0.getDriverExecTime());
      System.out.println("driver_memcopy_from_device_time: "+row0.getDriverMemcopyFromDeviceTime());
      System.out.println("total_driver_execution_time: "+row0.getTotalDriverExecutionTime());
      System.out.println("deserialization_time: "+row0.getDeserializationTime());
      System.out.println("gpu_required_memory: "+context0.getRequiredMemory());
      System.out.println("gpu_time: "+gpuTime);
      
      // check that array was properly sorted
      // populate with new random numbers
      for(int i = 0; i < outerCount; ++i){
        checkSorted(array[i], i);
        fisherYates(array[i]);
      }
      // sort on CPU as reference for comparison 
      long cpuStart = System.currentTimeMillis();
      for(int i = 0; i < outerCount; ++i){
        Arrays.sort(array[i]);
      }
      long cpuStop = System.currentTimeMillis();
      long cpuTime = cpuStop - cpuStart;
      System.out.println("cpu_time: "+cpuTime);
      double ratio = (double) cpuTime / (double) gpuTime;
      System.out.println("ratio: "+ratio);
//      System.out.println(Arrays.deepToString(array));
      ratioList.add(ratio);


    }
    double averageRatio = 0;
    for (double ratio : ratioList){
    	averageRatio += ratio;
    }
    System.out.println("Number of runs: " + runs);
    System.out.println("Average ratio of GPU<->CPU: "+averageRatio/runs);
//    context0.close();
  }

  public static void main(String[] args){
    GPUSort sorter = new GPUSort();
    sorter.sort();
//    while(true){
//      sorter.sort();
//    }
  }
}
