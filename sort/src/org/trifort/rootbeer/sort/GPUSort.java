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
      int index = random.nextInt(i + 1);
      int a = array[index];
      array[index] = array[i];
      array[i] = a;
    }
  }

  public void sort(){
    //should have 192 threads per SM
	// inner array size
    int size = 2048;
    int sizeBy2 = size / 2;
    //int numMultiProcessors = 14;
    //int blocksPerMultiProcessor = 512;
    int numMultiProcessors = 2;
    int blocksPerMultiProcessor = 256;

    int outerCount = numMultiProcessors*blocksPerMultiProcessor;
    int[][] array = new int[outerCount][];
    for(int i = 0; i < outerCount; ++i){
      array[i] = newArray(size);
    }

    Rootbeer rootbeer = new Rootbeer();
    List<GpuDevice> devices = rootbeer.getDevices();
    GpuDevice device0 = devices.get(0);
    Context context0 = device0.createContext(4212880);
    context0.setCacheConfig(CacheConfig.PREFER_SHARED);
    context0.setThreadConfig(sizeBy2, outerCount, outerCount * sizeBy2);
    context0.setKernel(new GPUSortKernel(array));
    context0.buildState();

    // limit the run
    int counter = 0;
    while(counter < 10){
      counter += 1;

//  while(true){
      // fill the array
      for(int i = 0; i < outerCount; ++i){
        fisherYates(array[i]);
      }
//      System.out.println(Arrays.deepToString(array));
      long gpuStart = System.currentTimeMillis();
      // sort the arrays
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


    }
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
