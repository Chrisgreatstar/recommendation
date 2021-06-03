import java.io.IOException;

class FISM_rmse {
    public FISM_rmse() throws IOException {
        
    }
}


class solution {
    public static void main(String[] args) throws IOException {
        long startTime=System.currentTimeMillis();
        new FISM_rmse();
		long endTime=System.currentTimeMillis();
		System.out.println("cost time: "+(endTime-startTime)+"ms");
    }
}