#include <stdio.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


typedef struct vertex vertex;

struct vertex {
    unsigned int vertex_id;
    float pagerank;
    float pagerank_next;
    unsigned int n_successors;
    vertex ** successors;
};


float abs_float(float in) {
  if (in >= 0)
    return in;
  else
    return -in;
}

__global__ void initializePagerankArray(float * pagerank_d, int n_vertices) {
    int i = threadIdx.x;

    pagerank_d[i] = 1.0/(float)n_vertices;
}

__global__ void setPagerankNextArray(float * pagerank_next_d) {
    int i = threadIdx.x;

    pagerank_next_d[i] = 0.0;
}


__global__ void addToNextPagerankArray(float * pagerank_d, float * pagerank_next_d, int * n_successors_d, int * successors_d, int * successor_offset_d, float * dangling_value2) {
    int i = threadIdx.x;
    int j;

    if(n_successors_d[i] > 0) {
        for(j = 0; j < n_successors_d[i]; j++) {
            atomicAdd(&(pagerank_next_d[successors_d[successor_offset_d[i]+j]]), 0.85*(pagerank_d[i])/n_successors_d[i]);
        }
    }else {
        atomicAdd(dangling_value2, 0.85*pagerank_d[i]);
    }
}       


__global__ void finalPagerankArrayForIteration(float * pagerank_next_d, int n_vertices, float dangling_value2) {
    int i = threadIdx.x;

    pagerank_next_d[i] += (dangling_value2 + (1-0.85))/((float)n_vertices);
}


__global__ void setPagerankArrayFromNext(float * pagerank_d, float * pagerank_next_d) {
    int i = threadIdx.x;

    pagerank_d[i] = pagerank_next_d[i];
    pagerank_next_d[i] = 0.0;
}      



int main(void) {
    // Error code to check return values for CUDA calls
    cudaFree(0);   // Set the cuda context here so that when we time, we're not including initial overhead
    cudaError_t err = cudaSuccess;
    

/*************************************************************************/
    // Start CPU timer
    clock_t start = clock(), diff;

/*************************************************************************/
    // build up the graph
    int i;//,j;
    unsigned int n_vertices = 0;
    unsigned int n_edges = 0;
    unsigned int vertex_from = 0, vertex_to = 0;

    vertex * vertices;
    
    // Flattened data structure variables
    float * pagerank_h, *pagerank_d;
    float *pagerank_next_d;
    int * n_successors_h, *n_successors_d;
    int * successors_h, *successors_d;                // use n_edges to initialize
    int * successor_offset_h, *successor_offset_d;

    FILE * fp;
    if ((fp = fopen("testInput.txt", "r")) == NULL) {
        fprintf(stderr,"ERROR: Could not open input file.\n");
        exit(-1);
    }

    // parse input file to count the number of vertices
    // expected format: vertex_from vertex_to
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        if (vertex_from > n_vertices)
            n_vertices = vertex_from;
        else if (vertex_to > n_vertices)
            n_vertices = vertex_to;
    }
    n_vertices++;
    

    // Allocate memory for vertices on host and device
    vertices = (vertex *) malloc(n_vertices * sizeof(vertex));
 
    // Allocate flattened data structure host and device memory
    pagerank_h = (float *) malloc(n_vertices * sizeof(*pagerank_h));
    err = cudaMalloc((void **)&pagerank_d, n_vertices*sizeof(float));
    err = cudaMalloc((void **)&pagerank_next_d, n_vertices*sizeof(float));
    n_successors_h = (int *) calloc(n_vertices, sizeof(*n_successors_h));
    err = cudaMalloc((void **)&n_successors_d, n_vertices*sizeof(int));
    successor_offset_h = (int *) malloc(n_vertices * sizeof(*successor_offset_h));
    err = cudaMalloc((void **)&successor_offset_d, n_vertices*sizeof(int));


    // SET Initial Parameters  **********************************************************
    /*  
    unsigned int n_iterations = 25;
    float alpha = 0.85;
    float eps   = 0.000001;
    */
   

    if (!vertices) {
        fprintf(stderr,"Malloc failed for vertices.\n");
        exit(-1);
    }
    memset((void *)vertices, 0, (size_t)(n_vertices*sizeof(vertex)));

    // parse input file to count the number of successors of each vertex
    fseek(fp, 0L, SEEK_SET);
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        vertices[vertex_from].n_successors++;
        n_successors_h[vertex_from] += 1;
        n_edges++;
    }

    // Allocate memory for contiguous successors_d data
    successors_h = (int *) malloc(n_edges * sizeof(*successors_h));
    err = cudaMalloc((void **)&successors_d, n_edges*sizeof(int));


    // allocate memory for successor pointers
    int offset = 0;                                         // offset into the successors_h array

    for (i=0; i<n_vertices; i++) {
        //vertices[i].vertex_id = i;
        successor_offset_h[i] = offset;
    
        if (vertices[i].n_successors > 0) {
           /* vertices[i].successors = (vertex **) malloc(vertices[i].n_successors * sizeof(vertex *));
            
            if (!vertices[i].successors) {
                fprintf(stderr,"Malloc failed for successors of vertex %d.\n",i);
                exit(-1);
            }
            memset((void *)vertices[i].successors, 0, (size_t)(vertices[i].n_successors*sizeof(vertex *)));*/
            offset += vertices[i].n_successors;
        }
        //else
            // vertices[i].successors = NULL;
            // Flattened structure is basically a offset + 0 here
    }

    // parse input file to set up the successor pointers
    int suc_index = 0;                                             // index into successors_h array
    fseek(fp, 0L, SEEK_SET);
    while (fscanf(fp, "%d %d", &vertex_from, &vertex_to) != EOF) {
        // Flattened data structure code to fill successors_h
        successors_h[suc_index] = vertex_to;
        suc_index++;            
        
        /*        
        for (i=0; i<vertices[vertex_from].n_successors; i++) {
            if (vertices[vertex_from].successors[i] == NULL) {
                vertices[vertex_from].successors[i] = &vertices[vertex_to];
                break;
            }
            else if (i==vertices[vertex_from].n_successors-1) {
                printf("Setting up the successor pointers of virtex %u failed",vertex_from);
                return -1;
            }
        }
        */
    }

    fclose(fp);

/**************************************************************/
    // Transfer data structure to the GPU
    err = cudaMemcpy(n_successors_d, n_successors_h, n_vertices*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(successors_d, successors_h, n_edges*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(successor_offset_d, successor_offset_h, n_vertices*sizeof(int), cudaMemcpyHostToDevice);

/*************************************************************************/
    // compute the pagerank on the GPU
    float dangling_value_h = 0;
    float dangling_value_h2 = 0;
    float *dangling_value2;
    
    err = cudaMalloc((void **)&dangling_value2, sizeof(float));
    err = cudaMemcpy(dangling_value2, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);

    initializePagerankArray<<<1,46>>>(pagerank_d, n_vertices);
    setPagerankNextArray<<<1, 46>>>(pagerank_next_d);
    cudaDeviceSynchronize();
    
    for(i = 0; i < 123; i++) {  //was 23
      //  setPagerankNextArray<<<1,46>>>(pagerank_next_d);
      //  cudaDeviceSynchronize();
       
        // set the dangling value to 0 
        dangling_value_h = 0;
        err = cudaMemcpy(dangling_value2, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);
       
 
        // initial parallel pagerank_next computation
        addToNextPagerankArray<<<1,46>>>(pagerank_d, pagerank_next_d, n_successors_d, successors_d, successor_offset_d, dangling_value2);
        cudaDeviceSynchronize();

        // get the dangling value
        err = cudaMemcpy(&dangling_value_h2, dangling_value2, sizeof(float), cudaMemcpyDeviceToHost); 

        // final parallel pagerank_next computation
        finalPagerankArrayForIteration<<<1,46>>>(pagerank_next_d, n_vertices, dangling_value_h2);
        cudaDeviceSynchronize();

        // Make pagerank_d[i] = pagerank_next_d[i]
        setPagerankArrayFromNext<<<1,46>>>(pagerank_d, pagerank_next_d);
        cudaDeviceSynchronize(); 
    }


/*****************************************************************************************/ 
  // Compute pagerank on host using old method for comparison purposes
  /*  unsigned int i_iteration;

    float value, diff;
    float pr_dangling_factor = alpha / (float)n_vertices;   // pagerank to redistribute from dangling nodes
    float pr_dangling;
    float pr_random_factor = (1-alpha) / (float)n_vertices; // random portion of the pagerank
    float pr_random;
    float pr_sum, pr_sum_inv, pr_sum_dangling;
    float temp;

    // initialization of values before pagerank loop
    for (i=0;i<n_vertices;i++) {
        vertices[i].pagerank = 1 / (float)n_vertices;
        vertices[i].pagerank_next =  0;
    }

    pr_sum = 0;
    pr_sum_dangling = 0;
    for (i=0; i<n_vertices; i++) {
        pr_sum += vertices[i].pagerank;
        if (!vertices[i].n_successors)
            pr_sum_dangling += vertices[i].pagerank;
    }

    i_iteration = 0;
    diff = eps+1;
*/
    //****** transfer pageranks from device to host memory ************************************************ 
    err = cudaMemcpy(pagerank_h, pagerank_d, n_vertices*sizeof(float), cudaMemcpyDeviceToHost);

    // Find CPU elapsed time
    diff = clock() - start;
    printf("diff: %Lf\n", (long double)diff); 
    // Print time taken
    int millisec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken: %d milliseconds\n", millisec%1000);
    
    // Print pageranks
    for(i = 0; i < n_vertices; i++) {
        printf("i: %d, pr: %.6f\n",i, pagerank_h[i]);
    }    

    //********************************************************************************************* 
  /*  while ( (diff > eps) && (i_iteration < n_iterations) ) {

        for (i=0;i<n_vertices;i++) {
            if (vertices[i].n_successors)
                value = (alpha/vertices[i].n_successors)*vertices[i].pagerank;  //value = vote value after splitting equally
            else
                value = 0;
            //printf("vertex %d: value = %.6f \n",i,value);
            for (j=0;j<vertices[i].n_successors;j++) {               // pagerank_next = sum of votes linking to it
                vertices[i].successors[j]->pagerank_next += value;
            }
        }
    
        // for normalization
        pr_sum_inv = 1/pr_sum;

        // alpha
        pr_dangling = pr_dangling_factor * pr_sum_dangling;
        pr_random = pr_random_factor * pr_sum;

        pr_sum = 0;
        pr_sum_dangling = 0;

        diff = 0;
        for (i=0;i<n_vertices;i++) {
            // update pagerank
            temp = vertices[i].pagerank;
            vertices[i].pagerank = vertices[i].pagerank_next*pr_sum_inv + pr_dangling + pr_random;
            vertices[i].pagerank_next = 0;

            // for normalization in next cycle
            pr_sum += vertices[i].pagerank;
            if (!vertices[i].n_successors)
                pr_sum_dangling += vertices[i].pagerank;

            // convergence
            diff += abs_float(temp - vertices[i].pagerank);
        }
        printf("Iteration %u:\t diff = %.12f\n", i_iteration, diff);

        i_iteration++;
    }
*/
/*************************************************************************/
    // print the pageranks from this host computation
  /*  for (i=0;i<n_vertices;i++) {
        printf("Vertex %u:\tpagerank = %.6f\n", i, vertices[i].pagerank);
    }
  */
/*************************************************************************/

    // Free device global memory
    err = cudaFree(pagerank_d);
    err = cudaFree(pagerank_next_d);

    // Free host memory
    free(pagerank_h);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

