
package org.trifort.rootbeer.examples.scalarsquare;

import org.trifort.rootbeer.runtime.Kernel;

public class ScalarSquareKernel implements Kernel {
	
  private int[] array;
  private int index;
	
  public ScalarSquareKernel(int[] array, int index){
    this.array = array;
    this.index = index;
  }

  public void gpuMethod(){
    array[index] = array[index] * array[index];
  }
}